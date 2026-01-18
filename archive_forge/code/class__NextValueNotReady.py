import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
class _NextValueNotReady(Exception):
    """Indicates that a local iterator has no value currently available.

    This is used internally to implement the union() of multiple blocking
    local generators."""
    pass