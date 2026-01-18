import os
from threading import Lock
import time
import types
from typing import (
import warnings
from . import values  # retain this import style for testability
from .context_managers import ExceptionCounter, InprogressTracker, Timer
from .metrics_core import (
from .registry import Collector, CollectorRegistry, REGISTRY
from .samples import Exemplar, Sample
from .utils import floatToGoString, INF
def _raise_if_not_observable(self):
    if not self._is_observable():
        raise ValueError('%s metric is missing label values' % str(self._type))