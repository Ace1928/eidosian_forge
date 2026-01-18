import logging
import os
import threading
from collections import deque
from typing import TYPE_CHECKING, List, Optional
from .aggregators import aggregate_mean
from .asset_registry import asset_registry
from .interfaces import Interface, Metric, MetricsMonitor
class TPUUtilization:
    """Google Cloud TPU utilization in percent."""
    name = 'tpu'
    samples: 'Deque[float]'

    def __init__(self, service_addr: str, duration_ms: int=100) -> None:
        self.samples = deque([])
        self.duration_ms = duration_ms
        self.service_addr = service_addr
        try:
            from tensorflow.python.profiler import profiler_client
            self._profiler_client = profiler_client
        except ImportError:
            logger.warning('Unable to import `tensorflow.python.profiler.profiler_client`. TPU metrics will not be reported.')
            self._profiler_client = None

    def sample(self) -> None:
        result = self._profiler_client.monitor(self.service_addr, duration_ms=self.duration_ms, level=2)
        self.samples.append(float(result.split('Utilization ')[1].split(': ')[1].split('%')[0]))

    def clear(self) -> None:
        self.samples.clear()

    def aggregate(self) -> dict:
        if not self.samples:
            return {}
        aggregate = aggregate_mean(self.samples)
        return {self.name: aggregate}