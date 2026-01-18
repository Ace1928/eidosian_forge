import inspect
import sys
from magnumclient.i18n import _
class AuthorizationFailure(ClientException):
    """Cannot authorize API client."""
    pass