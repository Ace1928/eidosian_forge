import gettext
import os
from oslo_i18n import _lazy
from oslo_i18n import _locale
from oslo_i18n import _message
def _make_log_translation_func(self, level):
    return self._make_translation_func(self.domain + '-log-' + level)