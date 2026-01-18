from __future__ import unicode_literals
import json
from oauthlib.common import add_params_to_uri, urlencode
class InsufficientScopeError(OAuth2Error):
    """
    The request requires higher privileges than provided by the
    access token.  The resource server SHOULD respond with the HTTP
    403 (Forbidden) status code and MAY include the "scope"
    attribute with the scope necessary to access the protected
    resource.
    """
    error = 'insufficient_scope'
    status_code = 403
    description = 'The request requires higher privileges than provided by the access token.'