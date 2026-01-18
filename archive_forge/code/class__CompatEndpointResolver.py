import boto.vendored.regions.regions as _regions
class _CompatEndpointResolver(_regions.EndpointResolver):
    """Endpoint resolver which handles boto2 compatibility concerns.

    This is NOT intended for external use whatsoever.
    """
    _DEFAULT_SERVICE_RENAMES = {'awslambda': 'lambda', 'cloudwatch': 'monitoring', 'ses': 'email', 'ec2containerservice': 'ecs', 'configservice': 'config'}

    def __init__(self, endpoint_data, service_rename_map=None):
        """
        :type endpoint_data: dict
        :param endpoint_data: Regions and endpoints data in the same format
            as is used by botocore / boto3.

        :type service_rename_map: dict
        :param service_rename_map: A mapping of boto2 service name to
            endpoint prefix.
        """
        super(_CompatEndpointResolver, self).__init__(endpoint_data)
        if service_rename_map is None:
            service_rename_map = self._DEFAULT_SERVICE_RENAMES
        self._endpoint_prefix_map = service_rename_map
        self._service_name_map = dict(((v, k) for k, v in service_rename_map.items()))

    def get_available_endpoints(self, service_name, partition_name='aws', allow_non_regional=False):
        endpoint_prefix = self._endpoint_prefix(service_name)
        return super(_CompatEndpointResolver, self).get_available_endpoints(endpoint_prefix, partition_name, allow_non_regional)

    def get_all_available_regions(self, service_name):
        """Retrieve every region across partitions for a service."""
        regions = set()
        endpoint_prefix = self._endpoint_prefix(service_name)
        for partition_name in self.get_available_partitions():
            if self._is_global_service(service_name, partition_name):
                partition = self._get_partition_data(partition_name)
                regions.update(partition['regions'].keys())
                continue
            else:
                regions.update(self.get_available_endpoints(endpoint_prefix, partition_name))
        return list(regions)

    def construct_endpoint(self, service_name, region_name=None):
        endpoint_prefix = self._endpoint_prefix(service_name)
        return super(_CompatEndpointResolver, self).construct_endpoint(endpoint_prefix, region_name)

    def get_available_services(self):
        """Get a list of all the available services in the endpoints file(s)"""
        services = set()
        for partition in self._endpoint_data['partitions']:
            services.update(partition['services'].keys())
        return [self._service_name(s) for s in services]

    def _is_global_service(self, service_name, partition_name='aws'):
        """Determines whether a service uses a global endpoint.

        In theory a service can be 'global' in one partition but regional in
        another. In practice, each service is all global or all regional.
        """
        endpoint_prefix = self._endpoint_prefix(service_name)
        partition = self._get_partition_data(partition_name)
        service = partition['services'].get(endpoint_prefix, {})
        return 'partitionEndpoint' in service

    def _get_partition_data(self, partition_name):
        """Get partition information for a particular partition.

        This should NOT be used to get service endpoint data because it only
        loads from the new endpoint format. It should only be used for
        partition metadata and partition specific service metadata.

        :type partition_name: str
        :param partition_name: The name of the partition to search for.

        :returns: Partition info from the new endpoints format.
        :rtype: dict or None
        """
        for partition in self._endpoint_data['partitions']:
            if partition['partition'] == partition_name:
                return partition
        raise ValueError('Could not find partition data for: %s' % partition_name)

    def _endpoint_prefix(self, service_name):
        """Given a boto2 service name, get the endpoint prefix."""
        return self._endpoint_prefix_map.get(service_name, service_name)

    def _service_name(self, endpoint_prefix):
        """Given an endpoint prefix, get the boto2 service name."""
        return self._service_name_map.get(endpoint_prefix, endpoint_prefix)