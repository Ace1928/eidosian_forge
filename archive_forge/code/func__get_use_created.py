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
def _get_use_created() -> bool:
    return os.environ.get('PROMETHEUS_DISABLE_CREATED_SERIES', 'False').lower() not in ('true', '1', 't')