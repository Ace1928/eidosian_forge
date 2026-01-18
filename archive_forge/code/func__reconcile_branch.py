from . import errors, ui
from .i18n import gettext
from .trace import mutter
def _reconcile_branch(self):
    try:
        self.branch = self.controldir.open_branch()
    except errors.NotBranchError:
        return
    ui.ui_factory.note(gettext('Reconciling branch %s') % self.branch.base)
    return self.branch.reconcile(thorough=True)