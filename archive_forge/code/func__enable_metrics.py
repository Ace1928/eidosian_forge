from oslo_log import log as logging
from os_win._i18n import _
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
def _enable_metrics(self, element, metrics_names=None):
    if not metrics_names:
        definition_paths = [None]
    else:
        definition_paths = []
        for metrics_name in metrics_names:
            metrics_def = self._metrics_defs.get(metrics_name)
            if not metrics_def:
                LOG.warning('Metric not found: %s', metrics_name)
                continue
            definition_paths.append(metrics_def.path_())
    element_path = element.path_()
    for definition_path in definition_paths:
        ret_val = self._metrics_svc.ControlMetrics(Subject=element_path, Definition=definition_path, MetricCollectionEnabled=self._METRICS_ENABLED)[0]
        if ret_val:
            err_msg = _('Failed to enable metrics for resource %(resource_name)s. Return code: %(ret_val)s.') % dict(resource_name=element.ElementName, ret_val=ret_val)
            raise exceptions.OSWinException(err_msg)