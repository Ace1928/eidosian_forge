import collections
from typing import List
from ray.util.annotations import Deprecated
from ray.util.timer import _Timer
@Deprecated
class SharedMetrics:
    """Holds an indirect reference to a (shared) metrics context.

    This is used by LocalIterator.union() to point the metrics contexts of
    entirely separate iterator chains to the same underlying context."""

    def __init__(self, metrics: MetricsContext=None, parents: List['SharedMetrics']=None):
        self.metrics = metrics or MetricsContext()
        self.parents = parents or []
        self.set(self.metrics)

    def set(self, metrics):
        """Recursively set self and parents to point to the same metrics."""
        self.metrics = metrics
        for parent in self.parents:
            parent.set(metrics)

    def get(self):
        return self.metrics