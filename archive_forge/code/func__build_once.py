import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def _build_once(self):
    if self.built_iterator is None:
        it = iter(self.base_iterator(self.timeout))
        for fn in self.local_transforms:
            it = fn(it)
        self.built_iterator = it