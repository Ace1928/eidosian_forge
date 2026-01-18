import boto.vendored.regions.regions as _regions
class BotoEndpointResolver(object):
    """Resolves endpoint hostnames for AWS services.

    This is NOT intended for external use.
    """

    def __init__(self, endpoint_data, service_rename_map=None):
        """
        :type endpoint_data: dict
        :param endpoint_data: Regions and endpoints data in the same format
            as is used by botocore / boto3.

        :type service_rename_map: dict
        :param service_rename_map: A mapping of boto2 service name to
            endpoint prefix.
        """
        self._resolver = _CompatEndpointResolver(endpoint_data, service_rename_map)

    def resolve_hostname(self, service_name, region_name):
        """Resolve the hostname for a service in a particular region.

        :type service_name: str
        :param service_name: The service to look up.

        :type region_name: str
        :param region_name: The region to find the endpoint for.

        :return: The hostname for the given service in the given region.
        """
        endpoint = self._resolver.construct_endpoint(service_name, region_name)
        if endpoint is None:
            return None
        return endpoint.get('sslCommonName', endpoint['hostname'])

    def get_all_available_regions(self, service_name):
        """Get all the regions a service is available in.

        :type service_name: str
        :param service_name: The service to look up.

        :rtype: list of str
        :return: A list of all the regions the given service is available in.
        """
        return self._resolver.get_all_available_regions(service_name)

    def get_available_services(self):
        """Get all the services supported by the endpoint data.

        :rtype: list of str
        :return: A list of all the services explicitly contained within the
            endpoint data provided during instantiation.
        """
        return self._resolver.get_available_services()