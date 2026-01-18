from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
class InsecureTransportError(OAuth2Error):
    error = 'insecure_transport'
    description = 'OAuth 2 MUST utilize https.'