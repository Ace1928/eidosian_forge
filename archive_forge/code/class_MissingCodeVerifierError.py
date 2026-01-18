from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
class MissingCodeVerifierError(InvalidRequestError):
    """
    The request to the token endpoint, when PKCE is enabled, has
    the parameter `code_verifier` REQUIRED.
    """
    description = 'Code verifier required.'