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
def _validate_exemplar(exemplar):
    runes = 0
    for k, v in exemplar.items():
        _validate_labelname(k)
        runes += len(k)
        runes += len(v)
    if runes > 128:
        raise ValueError('Exemplar labels have %d UTF-8 characters, exceeding the limit of 128')