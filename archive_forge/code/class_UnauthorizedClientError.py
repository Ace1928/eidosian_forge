from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
class UnauthorizedClientError(OAuth2Error):
    """
    The authenticated client is not authorized to use this authorization
    grant type.
    """
    error = 'unauthorized_client'