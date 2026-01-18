from oslo_context import context
from oslo_log import log as logging
import webob.exc
from heat.common.i18n import _
from heat.rpc import client as rpc_client
class AuthProtocol(object):

    def __init__(self, app, conf):
        self.conf = conf
        self.app = app
        self.rpc_client = rpc_client.EngineClient()

    def __call__(self, env, start_response):
        """Handle incoming request.

        Authenticate send downstream on success. Reject request if
        we can't authenticate.
        """
        LOG.debug('Authenticating user token')
        ctx = context.get_current()
        authenticated = self.rpc_client.authenticated_to_backend(ctx)
        if authenticated:
            return self.app(env, start_response)
        else:
            return self._reject_request(env, start_response)

    def _reject_request(self, env, start_response):
        """Redirect client to auth server.

        :param env: wsgi request environment
        :param start_response: wsgi response callback
        :returns: HTTPUnauthorized http response
        """
        resp = webob.exc.HTTPUnauthorized(_('Backend authentication failed'), [])
        return resp(env, start_response)