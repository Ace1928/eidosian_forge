from __future__ import absolute_import, unicode_literals
import time
import warnings
from oauthlib.common import generate_token
from oauthlib.oauth2.rfc6749 import tokens
from oauthlib.oauth2.rfc6749.errors import (InsecureTransportError,
from oauthlib.oauth2.rfc6749.parameters import (parse_token_response,
from oauthlib.oauth2.rfc6749.utils import is_secure_transport
def _add_bearer_token(self, uri, http_method='GET', body=None, headers=None, token_placement=None):
    """Add a bearer token to the request uri, body or authorization header."""
    if token_placement == AUTH_HEADER:
        headers = tokens.prepare_bearer_headers(self.access_token, headers)
    elif token_placement == URI_QUERY:
        uri = tokens.prepare_bearer_uri(self.access_token, uri)
    elif token_placement == BODY:
        body = tokens.prepare_bearer_body(self.access_token, body)
    else:
        raise ValueError('Invalid token placement.')
    return (uri, headers, body)