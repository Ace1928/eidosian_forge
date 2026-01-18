import abc
import urllib.parse
from keystoneauth1 import _utils as utils
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1.identity import base
@property
def _v2_params(self):
    """Return the parameters that are common to v2 plugins."""
    return {'trust_id': self._trust_id, 'tenant_id': self._project_id, 'tenant_name': self._project_name, 'reauthenticate': self.reauthenticate}