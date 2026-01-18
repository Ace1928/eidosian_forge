import os
from typing import Callable, Iterable, Optional, Union
from .metrics_core import CounterMetricFamily, GaugeMetricFamily, Metric
from .registry import Collector, CollectorRegistry, REGISTRY
def _boot_time(self):
    with open(os.path.join(self._proc, 'stat'), 'rb') as stat:
        for line in stat:
            if line.startswith(b'btime '):
                return float(line.split()[1])