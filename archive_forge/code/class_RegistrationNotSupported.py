from __future__ import unicode_literals
from oauthlib.oauth2.rfc6749.errors import FatalClientError, OAuth2Error
class RegistrationNotSupported(OpenIDClientError):
    """
    The OP does not support use of the registration parameter.
    """
    error = 'registration_not_supported'
    description = 'The registration parameter is not supported.'