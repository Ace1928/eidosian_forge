import logging
import time
from urllib.parse import parse_qs
from urllib.parse import urlencode
from urllib.parse import urlsplit
from saml2 import SAMLError
import saml2.cryptography.symmetric
from saml2.httputil import Redirect
from saml2.httputil import Response
from saml2.httputil import Unauthorized
from saml2.httputil import make_cookie
from saml2.httputil import parse_cookie
class AuthnMethodChooser:

    def __init__(self, methods=None):
        self.methods = methods

    def __call__(self, **kwargs):
        if not self.methods:
            raise SAMLError('No authentication methods defined')
        elif len(self.methods) == 1:
            return self.methods[0]
        else:
            pass