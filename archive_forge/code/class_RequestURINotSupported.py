from __future__ import unicode_literals
from oauthlib.oauth2.rfc6749.errors import FatalClientError, OAuth2Error
class RequestURINotSupported(OpenIDClientError):
    """
    The OP does not support use of the request_uri parameter.
    """
    error = 'request_uri_not_supported'
    description = 'The request_uri parameter is not supported.'