from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
class MismatchingStateError(OAuth2Error):
    error = 'mismatching_state'
    description = 'CSRF Warning! State not equal in request and response.'