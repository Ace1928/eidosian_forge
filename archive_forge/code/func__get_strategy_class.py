import urllib.parse
from keystoneauth1 import discover
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import plugin
from keystoneclient.v3 import client as v3_client
from keystonemiddleware.auth_token import _auth
from keystonemiddleware.auth_token import _exceptions as ksm_exceptions
from keystonemiddleware.i18n import _
def _get_strategy_class(self):
    if self._requested_auth_version:
        if not discover.version_match(_V3RequestStrategy.AUTH_VERSION, self._requested_auth_interface):
            self._LOG.info('A version other than v3 was requested: %s', self._requested_auth_interface)
        return _V3RequestStrategy
    for klass in _REQUEST_STRATEGIES:
        if self._adapter.get_endpoint(version=klass.AUTH_VERSION):
            self._LOG.debug('Auth Token confirmed use of %s apis', klass.AUTH_VERSION)
            return klass
    versions = ['v%d.%d' % s.AUTH_VERSION for s in _REQUEST_STRATEGIES]
    self._LOG.error('No attempted versions [%s] supported by server', ', '.join(versions))
    msg = _('No compatible apis supported by server')
    raise ksm_exceptions.ServiceError(msg)