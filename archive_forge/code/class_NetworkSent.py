import threading
from collections import deque
from typing import TYPE_CHECKING, List
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
class NetworkSent:
    """Network bytes sent."""
    name = 'network.sent'
    samples: 'Deque[float]'

    def __init__(self) -> None:
        self.samples = deque([])
        self.sent_init = psutil.net_io_counters().bytes_sent

    def sample(self) -> None:
        self.samples.append(psutil.net_io_counters().bytes_sent - self.sent_init)

    def clear(self) -> None:
        self.samples.clear()

    def aggregate(self) -> dict:
        if not self.samples:
            return {}
        aggregate = aggregate_mean(self.samples)
        return {self.name: aggregate}