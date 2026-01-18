from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
class InvalidScopeError(OAuth2Error):
    """
    The requested scope is invalid, unknown, or malformed, or
    exceeds the scope granted by the resource owner.

    https://tools.ietf.org/html/rfc6749#section-5.2
    """
    error = 'invalid_scope'