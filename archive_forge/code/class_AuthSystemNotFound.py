import inspect
import sys
from magnumclient.i18n import _
class AuthSystemNotFound(AuthorizationFailure):
    """User has specified an AuthSystem that is not installed."""

    def __init__(self, auth_system):
        super(AuthSystemNotFound, self).__init__(_('AuthSystemNotFound: %r') % auth_system)
        self.auth_system = auth_system