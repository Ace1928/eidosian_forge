import threading
from collections import deque
from typing import TYPE_CHECKING, List, Optional
from wandb.errors.term import termwarn
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
@asset_registry.register
class Disk:

    def __init__(self, interface: 'Interface', settings: 'SettingsStatic', shutdown_event: threading.Event) -> None:
        self.name = self.__class__.__name__.lower()
        self.settings = settings
        self.metrics: List[Metric] = [DiskUsagePercent(list(settings._stats_disk_paths or ['/'])), DiskUsage(list(settings._stats_disk_paths or ['/'])), DiskIn(), DiskOut()]
        self.metrics_monitor = MetricsMonitor(self.name, self.metrics, interface, settings, shutdown_event)

    @classmethod
    def is_available(cls) -> bool:
        """Return a new instance of the CPU metrics."""
        return psutil is not None

    def probe(self) -> dict:
        disk_paths = list(self.settings._stats_disk_paths or ['/'])
        disk_metrics = {}
        for disk_path in disk_paths:
            try:
                total = psutil.disk_usage(disk_path).total / 1024 / 1024 / 1024
                used = psutil.disk_usage(disk_path).used / 1024 / 1024 / 1024
                disk_metrics[disk_path] = {'total': total, 'used': used}
            except Exception as e:
                termwarn(f'Could not access disk path {disk_path}: {e}', repeat=False)
        return {self.name: disk_metrics}

    def start(self) -> None:
        self.metrics_monitor.start()

    def finish(self) -> None:
        self.metrics_monitor.finish()