import threading
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union
import wandb
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
@asset_registry.register
class IPU:

    def __init__(self, interface: 'Interface', settings: 'SettingsStatic', shutdown_event: threading.Event) -> None:
        self.name = self.__class__.__name__.lower()
        self.metrics: List[Metric] = [IPUStats(settings._stats_pid)]
        self.metrics_monitor = MetricsMonitor(self.name, self.metrics, interface, settings, shutdown_event)

    @classmethod
    def is_available(cls) -> bool:
        return gcipuinfo is not None

    def start(self) -> None:
        self.metrics_monitor.start()

    def finish(self) -> None:
        self.metrics_monitor.finish()

    def probe(self) -> dict:
        device_data = self.metrics[0]._gc_ipu_info.getDevices()
        device_count = len(device_data)
        devices = []
        for i, device in enumerate(device_data):
            device_metrics: Dict[str, str] = dict(device)
            devices.append({'id': device_metrics.get('id') or i, 'board ipu index': device_metrics.get('board ipu index'), 'board type': device_metrics.get('board type') or 'unknown'})
        return {self.name: {'device_count': device_count, 'devices': devices, 'vendor': 'Graphcore'}}