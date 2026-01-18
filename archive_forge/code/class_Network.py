import threading
from collections import deque
from typing import TYPE_CHECKING, List
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
@asset_registry.register
class Network:

    def __init__(self, interface: 'Interface', settings: 'SettingsStatic', shutdown_event: threading.Event) -> None:
        self.name = self.__class__.__name__.lower()
        self.metrics: List[Metric] = [NetworkSent(), NetworkRecv()]
        self.metrics_monitor = MetricsMonitor(self.name, self.metrics, interface, settings, shutdown_event)

    def start(self) -> None:
        self.metrics_monitor.start()

    def finish(self) -> None:
        self.metrics_monitor.finish()

    @classmethod
    def is_available(cls) -> bool:
        """Return a new instance of the CPU metrics."""
        return psutil is not None

    def probe(self) -> dict:
        """Return a dict of the hardware information."""
        return {}