from . import branch, controldir, errors, trace, ui, urlutils
from .i18n import gettext
class AlreadyWithTrees(BzrDirError):
    _fmt = "Shared repository '%(display_url)s' already creates working trees."