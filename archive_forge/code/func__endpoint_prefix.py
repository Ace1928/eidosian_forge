import boto.vendored.regions.regions as _regions
def _endpoint_prefix(self, service_name):
    """Given a boto2 service name, get the endpoint prefix."""
    return self._endpoint_prefix_map.get(service_name, service_name)