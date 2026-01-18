import base64
import hashlib
import ssl
import webob
from oslo_log import log as logging
from oslo_serialization import jsonutils
from keystonemiddleware.auth_token import _user_plugin
from keystonemiddleware.auth_token import AuthProtocol
from keystonemiddleware import exceptions
from keystonemiddleware.i18n import _
def _is_valid_access_token(self, request):
    """Check the OAuth2.0 certificate-bound access token.

        :param request: Incoming request
        :rtype: if the access token is valid
        """
    try:
        wsgi_input = request.environ.get('wsgi.input')
        if not wsgi_input:
            self.log.warn('Unable to obtain the client certificate.')
            return False
        sock = wsgi_input.get_socket()
        if not sock:
            self.log.warn('Unable to obtain the client certificate.')
            return False
        peer_cert = sock.getpeercert(binary_form=True)
        if not peer_cert:
            self.log.warn('Unable to obtain the client certificate.')
            return False
    except Exception as error:
        self.log.warn('Unable to obtain the client certificate. %s' % str(error))
        return False
    access_token = None
    if request.authorization and request.authorization.authtype == 'Bearer':
        access_token = request.authorization.params
    if not access_token:
        self.log.info('Unable to obtain the token.')
        return False
    try:
        token_data, user_auth_ref = self._do_fetch_token(access_token, allow_expired=False)
        self._validate_token(user_auth_ref, allow_expired=False)
        token = token_data.get('token')
        oauth2_cred = token.get('oauth2_credential')
        if not oauth2_cred:
            self.log.info('Invalid OAuth2.0 certificate-bound access token: The token is not an OAuth2.0 credential access token.')
            return False
        token_thumb = oauth2_cred.get('x5t#S256')
        if self._confirm_certificate_thumbprint(token_thumb, peer_cert):
            self._confirm_token_bind(user_auth_ref, request)
            request.token_info = token_data
            request.token_auth = _user_plugin.UserAuthPlugin(user_auth_ref, None)
            return True
        else:
            self.log.info('Invalid OAuth2.0 certificate-bound access token: the access token dose not match the client certificate.')
            return False
    except exceptions.KeystoneMiddlewareException as err:
        self.log.info('Invalid OAuth2.0 certificate-bound access token: %s' % str(err))
        return False