from . import branch, controldir, errors, trace, ui, urlutils
from .i18n import gettext
def _plan_changes(self, want_tree, want_branch, want_bound, want_reference):
    """Determine which changes are needed to assume the configuration"""
    if not want_branch and (not want_reference):
        raise ReconfigurationNotSupported(self.controldir)
    if want_branch and want_reference:
        raise ReconfigurationNotSupported(self.controldir)
    if self.repository is None:
        if not want_reference:
            self._create_repository = True
    elif want_reference and self.repository.user_url == self.controldir.user_url:
        if not self.repository.is_shared():
            self._destroy_repository = True
    if self.referenced_branch is None:
        if want_reference:
            self._create_reference = True
            if self.local_branch is not None:
                self._destroy_branch = True
    elif not want_reference:
        self._destroy_reference = True
    if self.local_branch is None:
        if want_branch is True:
            self._create_branch = True
            if want_bound:
                self._bind = True
    elif want_bound:
        if self.local_branch.get_bound_location() is None:
            self._bind = True
    elif self.local_branch.get_bound_location() is not None:
        self._unbind = True
    if not want_tree and self.tree is not None:
        self._destroy_tree = True
    if want_tree and self.tree is None:
        self._create_tree = True