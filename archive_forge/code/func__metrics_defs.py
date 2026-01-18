from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
@property
def _metrics_defs(self):
    if not self._metrics_defs_obj:
        self._cache_metrics_defs()
    return self._metrics_defs_obj