from . import branch, controldir, errors, trace, ui, urlutils
from .i18n import gettext
class AlreadyLightweightCheckout(BzrDirError):
    _fmt = "'%(display_url)s' is already a lightweight checkout."