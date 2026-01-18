from . import errors, ui
from .i18n import gettext
from .trace import mutter
def _reconcile_repository(self):
    self.repo = self.controldir.find_repository()
    ui.ui_factory.note(gettext('Reconciling repository %s') % self.repo.user_url)
    self.pb.update(gettext('Reconciling repository'), 0, 1)
    if self.canonicalize_chks:
        try:
            self.repo.reconcile_canonicalize_chks
        except AttributeError:
            raise errors.BzrError(gettext('%s cannot canonicalize CHKs.') % (self.repo,))
        reconcile_result = self.repo.reconcile_canonicalize_chks()
    else:
        reconcile_result = self.repo.reconcile(thorough=True)
    if reconcile_result.aborted:
        ui.ui_factory.note(gettext('Reconcile aborted: revision index has inconsistent parents.'))
        ui.ui_factory.note(gettext('Run "brz check" for more details.'))
    else:
        ui.ui_factory.note(gettext('Reconciliation complete.'))
    return reconcile_result