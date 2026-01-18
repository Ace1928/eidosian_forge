from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
class UnsupportedCodeChallengeMethodError(InvalidRequestError):
    """
    If the server supporting PKCE does not support the requested
    transformation, the authorization endpoint MUST return the
    authorization error response with "error" value set to
    "invalid_request".  The "error_description" or the response of
    "error_uri" SHOULD explain the nature of error, e.g., transform
    algorithm not supported.
    """
    description = 'Transform algorithm not supported.'