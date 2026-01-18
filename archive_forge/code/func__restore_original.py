import gettext
import fixtures
from oslo_i18n import _lazy
from oslo_i18n import _message
def _restore_original(self):
    _lazy.enable_lazy(self._original_value)