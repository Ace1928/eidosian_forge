from ... import errors, trace, transport
from ...config import AuthenticationConfig, GlobalStack
from ...i18n import gettext
class UnknownLaunchpadUsername(errors.BzrError):
    _fmt = 'The user name %(user)s is not registered on Launchpad.'