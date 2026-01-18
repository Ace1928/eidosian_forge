from __future__ import unicode_literals
from oauthlib.oauth2.rfc6749.errors import FatalClientError, OAuth2Error
class RequestNotSupported(OpenIDClientError):
    """
    The OP does not support use of the request parameter.
    """
    error = 'request_not_supported'
    description = 'The request parameter is not supported.'