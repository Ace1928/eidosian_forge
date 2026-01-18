from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
def _sum_metrics_values_by_defs(self, element_metrics, metrics_defs):
    metrics_values = []
    for metrics_def in metrics_defs:
        if metrics_def:
            metrics = self._filter_metrics(element_metrics, metrics_def)
            metrics_values.append(self._sum_metrics_values(metrics))
        else:
            metrics_values.append(0)
    return metrics_values