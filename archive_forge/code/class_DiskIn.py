import threading
from collections import deque
from typing import TYPE_CHECKING, List, Optional
from wandb.errors.term import termwarn
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
class DiskIn:
    """Total system disk read in MB."""
    name = 'disk.in'
    samples: 'Deque[float]'

    def __init__(self) -> None:
        self.samples = deque([])
        self.read_init: Optional[int] = None

    def sample(self) -> None:
        if self.read_init is None:
            self.read_init = psutil.disk_io_counters().read_bytes
        self.samples.append((psutil.disk_io_counters().read_bytes - self.read_init) / 1024 / 1024)

    def clear(self) -> None:
        self.samples.clear()

    def aggregate(self) -> dict:
        if not self.samples:
            return {}
        aggregate = aggregate_mean(self.samples)
        return {self.name: aggregate}