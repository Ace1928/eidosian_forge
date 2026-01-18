from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
class AccessDeniedError(OAuth2Error):
    """
    The resource owner or authorization server denied the request.
    """
    error = 'access_denied'