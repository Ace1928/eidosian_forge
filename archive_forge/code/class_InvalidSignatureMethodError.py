from __future__ import unicode_literals
from oauthlib.common import add_params_to_uri, urlencode
class InvalidSignatureMethodError(OAuth1Error):
    error = 'invalid_signature_method'