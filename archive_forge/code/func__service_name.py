import boto.vendored.regions.regions as _regions
def _service_name(self, endpoint_prefix):
    """Given an endpoint prefix, get the boto2 service name."""
    return self._service_name_map.get(endpoint_prefix, endpoint_prefix)