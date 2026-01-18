from __future__ import unicode_literals
from oauthlib.oauth2.rfc6749.errors import FatalClientError, OAuth2Error
class InvalidRequestURI(OpenIDClientError):
    """
    The request_uri in the Authorization Request returns an error or
    contains invalid data.
    """
    error = 'invalid_request_uri'
    description = 'The request_uri in the Authorization Request returns an error or contains invalid data.'