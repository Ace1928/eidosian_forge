import abc
import warnings
from keystoneclient import exceptions
from keystoneclient.i18n import _
def _is_endpoint_type_match(self, endpoint, endpoint_type):
    try:
        return endpoint_type == endpoint['interface']
    except KeyError:
        return False