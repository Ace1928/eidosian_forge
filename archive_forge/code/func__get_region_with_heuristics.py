import os
import boto
from boto.compat import json
from boto.exception import BotoClientError
from boto.endpoints import BotoEndpointResolver
from boto.endpoints import StaticEndpointBuilder
def _get_region_with_heuristics(service_name, region_name, region_cls=None, connection_cls=None):
    """Finds the region using known regions and heuristics."""
    endpoints = load_endpoint_json(boto.ENDPOINTS_PATH)
    resolver = BotoEndpointResolver(endpoints)
    hostname = resolver.resolve_hostname(service_name, region_name)
    return region_cls(name=region_name, endpoint=hostname, connection_cls=connection_cls)