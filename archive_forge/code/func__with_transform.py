import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def _with_transform(self, local_it_fn, name):
    """Helper function to create new Parallel Iterator"""
    return ParallelIterator([a.with_transform(local_it_fn) for a in self.actor_sets], name=self.name + name, parent_iterators=self.parent_iterators)