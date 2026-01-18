from collections import defaultdict, deque
import logging
import platform
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type
import ray
from ray.actor import ActorClass, ActorHandle
class TaskPool:
    """Helper class for tracking the status of many in-flight actor tasks."""

    def __init__(self):
        self._tasks = {}
        self._objects = {}
        self._fetching = deque()

    def add(self, worker, all_obj_refs):
        if isinstance(all_obj_refs, list):
            obj_ref = all_obj_refs[0]
        else:
            obj_ref = all_obj_refs
        self._tasks[obj_ref] = worker
        self._objects[obj_ref] = all_obj_refs

    def completed(self, blocking_wait=False):
        pending = list(self._tasks)
        if pending:
            ready, _ = ray.wait(pending, num_returns=len(pending), timeout=0)
            if not ready and blocking_wait:
                ready, _ = ray.wait(pending, num_returns=1, timeout=10.0)
            for obj_ref in ready:
                yield (self._tasks.pop(obj_ref), self._objects.pop(obj_ref))

    def completed_prefetch(self, blocking_wait=False, max_yield=999):
        """Similar to completed but only returns once the object is local.

        Assumes obj_ref only is one id."""
        for worker, obj_ref in self.completed(blocking_wait=blocking_wait):
            self._fetching.append((worker, obj_ref))
        for _ in range(max_yield):
            if not self._fetching:
                break
            yield self._fetching.popleft()

    def reset_workers(self, workers):
        """Notify that some workers may be removed."""
        for obj_ref, ev in self._tasks.copy().items():
            if ev not in workers:
                del self._tasks[obj_ref]
                del self._objects[obj_ref]
        for _ in range(len(self._fetching)):
            ev, obj_ref = self._fetching.popleft()
            if ev in workers:
                self._fetching.append((ev, obj_ref))

    @property
    def count(self):
        return len(self._tasks)