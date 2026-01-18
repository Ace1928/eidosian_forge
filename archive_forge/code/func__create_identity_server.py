from the request environment and it's identified by the ``swift.cache`` key.
import copy
import re
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import loading
from keystoneauth1.loading import session as session_loading
import oslo_cache
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
import webob.dec
from keystonemiddleware._common import config
from keystonemiddleware.auth_token import _auth
from keystonemiddleware.auth_token import _base
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.auth_token import _exceptions as ksm_exceptions
from keystonemiddleware.auth_token import _identity
from keystonemiddleware.auth_token import _opts
from keystonemiddleware.auth_token import _request
from keystonemiddleware.auth_token import _user_plugin
from keystonemiddleware.i18n import _
def _create_identity_server(self):
    adap = adapter.Adapter(self._session, auth=self._auth, service_type='identity', interface=self._interface, region_name=self._conf.get('region_name'), connect_retries=self._conf.get('http_request_max_retries'))
    auth_version = self._conf.get('auth_version')
    if auth_version is not None:
        auth_version = discover.normalize_version_number(auth_version)
    return _identity.IdentityServer(self.log, adap, include_service_catalog=self._include_service_catalog, requested_auth_version=auth_version, requested_auth_interface=self._interface)