from . import branch, controldir, errors, trace, ui, urlutils
from .i18n import gettext
class UnsyncedBranches(BzrDirError):
    _fmt = "'%(display_url)s' is not in sync with %(target_url)s.  See brz help sync-for-reconfigure."

    def __init__(self, controldir, target_branch):
        errors.BzrError.__init__(self, controldir)
        from . import urlutils
        self.target_url = urlutils.unescape_for_display(target_branch.base, 'ascii')