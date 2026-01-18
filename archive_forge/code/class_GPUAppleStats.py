import json
import logging
import pathlib
import platform
import subprocess
import sys
import threading
from collections import deque
from typing import TYPE_CHECKING, List
from wandb.sdk.lib import telemetry
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
class GPUAppleStats:
    """Apple GPU stats available on Arm Macs."""
    name = 'gpu.0.{}'
    samples: 'Deque[_Stats]'
    MAX_POWER_WATTS = 16.5

    def __init__(self) -> None:
        self.samples = deque()
        self.binary_path = (pathlib.Path(sys.modules['wandb'].__path__[0]) / 'bin' / 'apple_gpu_stats').resolve()

    def sample(self) -> None:
        try:
            command = [str(self.binary_path), '--json']
            output = subprocess.check_output(command, universal_newlines=True).strip().split('\n')[0]
            raw_stats = json.loads(output)
            stats: _Stats = {'gpu': raw_stats['utilization'], 'memoryAllocated': raw_stats['mem_used'], 'temp': raw_stats['temperature'], 'powerWatts': raw_stats['power'], 'powerPercent': raw_stats['power'] / self.MAX_POWER_WATTS * 100}
            self.samples.append(stats)
        except (OSError, ValueError, TypeError, subprocess.CalledProcessError) as e:
            logger.exception(f'GPU stats error: {e}')

    def clear(self) -> None:
        self.samples.clear()

    def aggregate(self) -> dict:
        if not self.samples:
            return {}
        stats = {}
        for key in self.samples[0].keys():
            samples = [s[key] for s in self.samples]
            aggregate = aggregate_mean(samples)
            stats[self.name.format(key)] = aggregate
        return stats