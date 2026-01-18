from typing import TYPE_CHECKING, Any, Callable, List, TypeVar
import ray
from ray.util.annotations import DeveloperAPI
def _return_actor(self, actor):
    self._idle_actors.append(actor)
    if self._pending_submits:
        self.submit(*self._pending_submits.pop(0))