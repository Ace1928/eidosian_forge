from ... import errors, trace, transport
from ...config import AuthenticationConfig, GlobalStack
from ...i18n import gettext
class NoRegisteredSSHKeys(errors.BzrError):
    _fmt = 'The user %(user)s has not registered any SSH keys with Launchpad.\nSee <https://launchpad.net/people/+me>'