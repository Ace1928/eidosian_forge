from ... import errors, trace, transport
from ...config import AuthenticationConfig, GlobalStack
from ...i18n import gettext
Check whether the given Launchpad username is okay.

    This will check for both existence and whether the user has
    uploaded SSH keys.
    