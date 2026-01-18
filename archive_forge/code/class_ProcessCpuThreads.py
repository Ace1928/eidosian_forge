import threading
from collections import deque
from typing import TYPE_CHECKING, List, Optional
from .aggregators import aggregate_last, aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
class ProcessCpuThreads:
    """Number of threads used by the process."""
    name = 'proc.cpu.threads'

    def __init__(self, pid: int) -> None:
        self.samples: Deque[int] = deque([])
        self.pid = pid
        self.process: Optional[psutil.Process] = None

    def sample(self) -> None:
        if self.process is None:
            self.process = psutil.Process(self.pid)
        self.samples.append(self.process.num_threads())

    def clear(self) -> None:
        self.samples.clear()

    def aggregate(self) -> dict:
        if not self.samples:
            return {}
        return {self.name: aggregate_last(self.samples)}