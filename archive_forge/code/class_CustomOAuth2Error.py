from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
class CustomOAuth2Error(OAuth2Error):
    """
    This error is a placeholder for all custom errors not described by the RFC.
    Some of the popular OAuth2 providers are using custom errors.
    """

    def __init__(self, error, *args, **kwargs):
        self.error = error
        super(CustomOAuth2Error, self).__init__(*args, **kwargs)