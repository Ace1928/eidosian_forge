from abc import ABC, abstractmethod
import copy
from threading import Lock
from typing import Dict, Iterable, List, Optional
from .metrics_core import Metric
def _target_info_metric(self):
    m = Metric('target', 'Target metadata', 'info')
    m.add_sample('target_info', self._target_info, 1)
    return m