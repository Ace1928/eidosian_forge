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
def _validate_labelnames(cls, labelnames):
    labelnames = tuple(labelnames)
    for l in labelnames:
        _validate_labelname(l)
        if l in cls._reserved_labelnames:
            raise ValueError('Reserved label metric name: ' + l)
    return labelnames