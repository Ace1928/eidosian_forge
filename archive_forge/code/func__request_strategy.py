import urllib.parse
from keystoneauth1 import discover
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import plugin
from keystoneclient.v3 import client as v3_client
from keystonemiddleware.auth_token import _auth
from keystonemiddleware.auth_token import _exceptions as ksm_exceptions
from keystonemiddleware.i18n import _
@property
def _request_strategy(self):
    if not self._request_strategy_obj:
        strategy_class = self._get_strategy_class()
        self._adapter.version = strategy_class.AUTH_VERSION
        self._request_strategy_obj = strategy_class(self._adapter, include_service_catalog=self._include_service_catalog, requested_auth_interface=self._requested_auth_interface)
    return self._request_strategy_obj