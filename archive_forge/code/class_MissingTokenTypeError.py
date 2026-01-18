from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
class MissingTokenTypeError(OAuth2Error):
    error = 'missing_token_type'