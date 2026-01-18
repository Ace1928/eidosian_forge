from ... import errors, trace, transport
from ...config import AuthenticationConfig, GlobalStack
from ...i18n import gettext
class MismatchedUsernames(errors.BzrError):
    _fmt = 'breezy.conf and authentication.conf disagree about launchpad account name.  Please re-run launchpad-login.'