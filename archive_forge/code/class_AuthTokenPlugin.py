from keystoneauth1 import discover
from keystoneauth1.identity import v2
from keystoneauth1 import plugin
from keystoneauth1 import token_endpoint
from oslo_config import cfg
from keystonemiddleware.auth_token import _base
from keystonemiddleware.i18n import _
class AuthTokenPlugin(plugin.BaseAuthPlugin):

    def __init__(self, auth_host, auth_port, auth_protocol, auth_admin_prefix, admin_user, admin_password, admin_tenant_name, admin_token, identity_uri, log):
        log.warning('Use of the auth_admin_prefix, auth_host, auth_port, auth_protocol, identity_uri, admin_token, admin_user, admin_password, and admin_tenant_name configuration options was deprecated in the Mitaka release in favor of an auth_plugin and its related options. This class may be removed in a future release.')
        if not identity_uri:
            log.warning("Configuring admin URI using auth fragments was deprecated in the Kilo release, and will be removed in the Newton release, use 'identity_uri' instead.")
            if ':' in auth_host:
                auth_host = '[%s]' % auth_host
            identity_uri = '%s://%s:%s' % (auth_protocol, auth_host, auth_port)
            if auth_admin_prefix:
                identity_uri = '%s/%s' % (identity_uri, auth_admin_prefix.strip('/'))
        self._identity_uri = identity_uri.rstrip('/')
        auth_url = '%s/v2.0' % self._identity_uri
        if admin_token:
            log.warning('The admin_token option in auth_token middleware was deprecated in the Kilo release, and will be removed in the Newton release, use admin_user and admin_password instead.')
            self._plugin = token_endpoint.Token(auth_url, admin_token)
        else:
            self._plugin = v2.Password(auth_url, username=admin_user, password=admin_password, tenant_name=admin_tenant_name)
        self._LOG = log
        self._discover = None

    def get_token(self, *args, **kwargs):
        return self._plugin.get_token(*args, **kwargs)

    def get_endpoint(self, session, interface=None, version=None, **kwargs):
        """Return an endpoint for the client.

        There are no required keyword arguments to ``get_endpoint`` as a plugin
        implementation should use best effort with the information available to
        determine the endpoint.

        :param session: The session object that the auth_plugin belongs to.
        :type session: keystoneauth1.session.Session
        :param version: The version number required for this endpoint.
        :type version: tuple or str
        :param str interface: what visibility the endpoint should have.

        :returns: The base URL that will be used to talk to the required
                  service or None if not available.
        :rtype: string
        """
        if interface == plugin.AUTH_INTERFACE:
            return self._identity_uri
        if not version:
            return None
        if not self._discover:
            self._discover = discover.Discover(session, url=self._identity_uri, authenticated=False)
        if not self._discover.url_for(version):
            return None
        version = discover.normalize_version_number(version)
        if discover.version_match((2, 0), version):
            return '%s/v2.0' % self._identity_uri
        elif discover.version_match((3, 0), version):
            return '%s/v3' % self._identity_uri
        msg = _('Invalid version asked for in auth_token plugin')
        raise NotImplementedError(msg)

    def invalidate(self):
        return self._plugin.invalidate()