import threading
from collections import deque
from typing import TYPE_CHECKING, List, Optional
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
class ProcessMemoryRSS:
    """Memory resident set size (RSS) in MB.

    RSS is the portion of memory occupied by a process that is held in main memory (RAM).
    """
    name = 'proc.memory.rssMB'
    samples: 'Deque[float]'

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.process: Optional[psutil.Process] = None
        self.samples = deque([])

    def sample(self) -> None:
        if self.process is None:
            self.process = psutil.Process(self.pid)
        self.samples.append(self.process.memory_info().rss / 1024 / 1024)

    def clear(self) -> None:
        self.samples.clear()

    def aggregate(self) -> dict:
        if not self.samples:
            return {}
        aggregate = aggregate_mean(self.samples)
        return {self.name: aggregate}