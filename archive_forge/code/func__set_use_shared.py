from . import branch, controldir, errors, trace, ui, urlutils
from .i18n import gettext
def _set_use_shared(self, use_shared=None):
    if use_shared is None:
        return
    if use_shared:
        if self.local_repository is not None:
            self._destroy_repository = True
    elif self.local_repository is None:
        self._create_repository = True