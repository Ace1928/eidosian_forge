import itertools
from oslo_serialization import jsonutils
import webob
def _normalize_catalog(catalog):
    """Convert a catalog to a compatible format."""
    services = []
    for v3_service in catalog:
        service = {'type': v3_service['type']}
        try:
            service['name'] = v3_service['name']
        except KeyError:
            pass
        regions = {}
        for v3_endpoint in v3_service.get('endpoints', []):
            region_name = v3_endpoint.get('region')
            try:
                region = regions[region_name]
            except KeyError:
                region = {'region': region_name} if region_name else {}
                regions[region_name] = region
            interface_name = v3_endpoint['interface'].lower() + 'URL'
            region[interface_name] = v3_endpoint['url']
        service['endpoints'] = list(regions.values())
        services.append(service)
    return services