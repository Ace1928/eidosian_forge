from __future__ import unicode_literals
from oauthlib.oauth2.rfc6749.errors import FatalClientError, OAuth2Error
class FatalOpenIDClientError(FatalClientError):
    pass