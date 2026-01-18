from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
class InvalidTokenError(OAuth2Error):
    """
    The access token provided is expired, revoked, malformed, or
    invalid for other reasons.  The resource SHOULD respond with
    the HTTP 401 (Unauthorized) status code.  The client MAY
    request a new access token and retry the protected resource
    request.
    """
    error = 'invalid_token'
    status_code = 401
    description = 'The access token provided is expired, revoked, malformed, or invalid for other reasons.'