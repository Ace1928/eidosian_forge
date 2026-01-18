from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
class UnsupportedGrantTypeError(OAuth2Error):
    """
    The authorization grant type is not supported by the authorization
    server.
    """
    error = 'unsupported_grant_type'