import boto.vendored.regions.regions as _regions
def _is_global_service(self, service_name, partition_name='aws'):
    """Determines whether a service uses a global endpoint.

        In theory a service can be 'global' in one partition but regional in
        another. In practice, each service is all global or all regional.
        """
    endpoint_prefix = self._endpoint_prefix(service_name)
    partition = self._get_partition_data(partition_name)
    service = partition['services'].get(endpoint_prefix, {})
    return 'partitionEndpoint' in service