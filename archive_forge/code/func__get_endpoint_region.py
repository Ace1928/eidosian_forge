import abc
import copy
from keystoneauth1 import discover
from keystoneauth1 import exceptions
def _get_endpoint_region(self, endpoint):
    return endpoint.get('region_id') or endpoint.get('region')