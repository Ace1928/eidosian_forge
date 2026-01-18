import logging
import re
from boto.vendored.regions.exceptions import NoRegionError
def _endpoint_for_partition(self, partition, service_name, region_name):
    service_data = partition['services'].get(service_name, DEFAULT_SERVICE_DATA)
    if region_name is None:
        if 'partitionEndpoint' in service_data:
            region_name = service_data['partitionEndpoint']
        else:
            raise NoRegionError()
    if region_name in service_data['endpoints']:
        return self._resolve(partition, service_name, service_data, region_name)
    if self._region_match(partition, region_name):
        partition_endpoint = service_data.get('partitionEndpoint')
        is_regionalized = service_data.get('isRegionalized', True)
        if partition_endpoint and (not is_regionalized):
            LOG.debug('Using partition endpoint for %s, %s: %s', service_name, region_name, partition_endpoint)
            return self._resolve(partition, service_name, service_data, partition_endpoint)
        LOG.debug('Creating a regex based endpoint for %s, %s', service_name, region_name)
        return self._resolve(partition, service_name, service_data, region_name)