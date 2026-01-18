import threading
from collections import deque
from typing import TYPE_CHECKING, List, Optional
from wandb.errors.term import termwarn
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
class DiskUsagePercent:
    """Total system disk usage in percent."""
    name = 'disk.{path}.usagePercent'
    samples: 'Deque[List[float]]'

    def __init__(self, paths: List[str]) -> None:
        self.samples = deque([])
        self.paths: List[str] = []
        for path in paths:
            try:
                psutil.disk_usage(path)
                self.paths.append(path)
            except Exception as e:
                termwarn(f'Could not access disk path {path}: {e}', repeat=False)

    def sample(self) -> None:
        disk_usage: List[float] = []
        for path in self.paths:
            disk_usage.append(psutil.disk_usage(path).percent)
        if disk_usage:
            self.samples.append(disk_usage)

    def clear(self) -> None:
        self.samples.clear()

    def aggregate(self) -> dict:
        if not self.samples:
            return {}
        disk_metrics = {}
        for i, _path in enumerate(self.paths):
            aggregate_i = aggregate_mean([sample[i] for sample in self.samples])
            _path = _path.replace('/', '\\')
            disk_metrics[self.name.format(path=_path)] = aggregate_i
        return disk_metrics