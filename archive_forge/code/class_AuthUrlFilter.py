from oslo_config import cfg
from webob import exc
from heat.common import endpoint_utils
from heat.common.i18n import _
from heat.common import wsgi
class AuthUrlFilter(wsgi.Middleware):

    def __init__(self, app, conf):
        super(AuthUrlFilter, self).__init__(app)
        self.conf = conf
        self._auth_url = None

    @property
    def auth_url(self):
        if not self._auth_url:
            self._auth_url = self._get_auth_url()
        return self._auth_url

    def _get_auth_url(self):
        if 'auth_uri' in self.conf:
            return self.conf['auth_uri']
        else:
            return endpoint_utils.get_auth_uri(v3=False)

    def _validate_auth_url(self, auth_url):
        """Validate auth_url to ensure it can be used."""
        if not auth_url:
            raise exc.HTTPBadRequest(_('Request missing required header X-Auth-Url'))
        allowed = cfg.CONF.auth_password.allowed_auth_uris
        if auth_url not in allowed:
            raise exc.HTTPUnauthorized(_('Header X-Auth-Url "%s" not an allowed endpoint') % auth_url)
        return True

    def process_request(self, req):
        auth_url = self.auth_url
        if cfg.CONF.auth_password.multi_cloud:
            auth_url = req.headers.get('X-Auth-Url')
            self._validate_auth_url(auth_url)
        req.headers['X-Auth-Url'] = auth_url
        return None