from typing import Any, List
import ray
from ray import cloudpickle
class RunningMean:

    def __init__(self):
        self._weight = 0
        self._mean = 0

    def add(self, x: int, weight: int=1) -> None:
        if weight == 0:
            return
        n1 = self._weight
        n2 = weight
        n = n1 + n2
        M = (n1 * self._mean + n2 * x) / n
        self._weight = n
        self._mean = M

    @property
    def n(self) -> int:
        return self._weight

    @property
    def mean(self) -> float:
        return self._mean

    def __repr__(self):
        return '(n={}, mean={})'.format(self.n, self.mean)