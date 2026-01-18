from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
class UnsupportedResponseTypeError(OAuth2Error):
    """
    The authorization server does not support obtaining an authorization
    code using this method.
    """
    error = 'unsupported_response_type'