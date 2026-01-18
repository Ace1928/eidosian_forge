from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
from triad import SerializableRLock, to_uuid
from tune.concepts.flow import (
class RungHeap:

    def __init__(self, n: int):
        self._lock = SerializableRLock()
        self._n = n
        self._heap = TrialReportHeap(min_heap=False)
        self._bests: List[float] = []

    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)

    @property
    def capacity(self) -> int:
        return self._n

    @property
    def best(self) -> float:
        with self._lock:
            return self._bests[-1] if len(self._bests) > 0 else float('nan')

    @property
    def bests(self) -> List[float]:
        with self._lock:
            return self._bests

    @property
    def full(self) -> bool:
        with self._lock:
            return self.capacity <= len(self)

    def __contains__(self, tid: str) -> bool:
        with self._lock:
            return tid in self._heap

    def values(self) -> Iterable[TrialReport]:
        return self._heap.values()

    def push(self, report: TrialReport) -> bool:
        with self._lock:
            if len(self) == 0:
                best = report.sort_metric
            else:
                best = min(self.best, report.sort_metric)
            self._heap.push(report)
            self._bests.append(best)
            return len(self._heap) <= self._n or self._heap.pop().trial_id != report.trial_id