import boto.vendored.regions.regions as _regions
def _build_endpoints_for_service(self, service_name):
    endpoints = {}
    regions = self._resolver.get_all_available_regions(service_name)
    for region_name in regions:
        endpoints[region_name] = self._resolver.resolve_hostname(service_name, region_name)
    return endpoints