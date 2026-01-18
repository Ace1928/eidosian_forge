from base64 import b64decode
from paste.httpexceptions import HTTPUnauthorized
from paste.httpheaders import (
class AuthBasicAuthenticator(object):
    """
    implements ``Basic`` authentication details
    """
    type = 'basic'

    def __init__(self, realm, authfunc):
        self.realm = realm
        self.authfunc = authfunc

    def build_authentication(self):
        head = WWW_AUTHENTICATE.tuples('Basic realm="%s"' % self.realm)
        return HTTPUnauthorized(headers=head)

    def authenticate(self, environ):
        authorization = AUTHORIZATION(environ)
        if not authorization:
            return self.build_authentication()
        authmeth, auth = authorization.split(' ', 1)
        if 'basic' != authmeth.lower():
            return self.build_authentication()
        auth = b64decode(auth.strip().encode('ascii')).decode('ascii')
        username, password = auth.split(':', 1)
        if self.authfunc(environ, username, password):
            return username
        return self.build_authentication()
    __call__ = authenticate