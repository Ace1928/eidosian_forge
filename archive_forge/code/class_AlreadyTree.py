from . import branch, controldir, errors, trace, ui, urlutils
from .i18n import gettext
class AlreadyTree(BzrDirError):
    _fmt = "'%(display_url)s' is already a tree."