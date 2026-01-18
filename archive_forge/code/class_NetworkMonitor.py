import psutil
from mlflow.system_metrics.metrics.base_metrics_monitor import BaseMetricsMonitor
class NetworkMonitor(BaseMetricsMonitor):

    def __init__(self):
        super().__init__()
        self._set_initial_metrics()

    def _set_initial_metrics(self):
        network_usage = psutil.net_io_counters()
        self._initial_receive_megabytes = network_usage.bytes_recv / 1000000.0
        self._initial_transmit_megabytes = network_usage.bytes_sent / 1000000.0

    def collect_metrics(self):
        network_usage = psutil.net_io_counters()
        self._metrics['network_receive_megabytes'] = network_usage.bytes_recv / 1000000.0 - self._initial_receive_megabytes
        self._metrics['network_transmit_megabytes'] = network_usage.bytes_sent / 1000000.0 - self._initial_transmit_megabytes

    def aggregate_metrics(self):
        return dict(self._metrics)