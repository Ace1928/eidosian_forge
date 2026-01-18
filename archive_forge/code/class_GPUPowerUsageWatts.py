import logging
import threading
from collections import deque
from typing import TYPE_CHECKING, List
from wandb.vendor.pynvml import pynvml
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
class GPUPowerUsageWatts:
    """GPU power usage in Watts for each GPU."""
    name = 'gpu.{}.powerWatts'
    samples: 'Deque[List[float]]'

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.samples = deque([])

    def sample(self) -> None:
        power_usage = []
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            power_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
            power_usage.append(power_watts)
        self.samples.append(power_usage)

    def clear(self) -> None:
        self.samples.clear()

    def aggregate(self) -> dict:
        stats = {}
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            samples = [sample[i] for sample in self.samples]
            aggregate = aggregate_mean(samples)
            stats[self.name.format(i)] = aggregate
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            if gpu_in_use_by_this_process(handle, self.pid):
                stats[self.name.format(f'process.{i}')] = aggregate
        return stats