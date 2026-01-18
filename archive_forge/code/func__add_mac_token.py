from __future__ import absolute_import, unicode_literals
import time
import warnings
from oauthlib.common import generate_token
from oauthlib.oauth2.rfc6749 import tokens
from oauthlib.oauth2.rfc6749.errors import (InsecureTransportError,
from oauthlib.oauth2.rfc6749.parameters import (parse_token_response,
from oauthlib.oauth2.rfc6749.utils import is_secure_transport
def _add_mac_token(self, uri, http_method='GET', body=None, headers=None, token_placement=AUTH_HEADER, ext=None, **kwargs):
    """Add a MAC token to the request authorization header.

        Warning: MAC token support is experimental as the spec is not yet stable.
        """
    headers = tokens.prepare_mac_header(self.access_token, uri, self.mac_key, http_method, headers=headers, body=body, ext=ext, hash_algorithm=self.mac_algorithm, **kwargs)
    return (uri, headers, body)