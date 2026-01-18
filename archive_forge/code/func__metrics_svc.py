from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
@property
def _metrics_svc(self):
    if not self._metrics_svc_obj:
        self._metrics_svc_obj = self._compat_conn.Msvm_MetricService()[0]
    return self._metrics_svc_obj