import functools
from keystoneauth1 import _utils as utils
from keystoneauth1.access import service_catalog
from keystoneauth1.access import service_providers
@property
def _trust(self):
    return self._data['token']['OS-TRUST:trust']