import logging
import random
import time
import uuid
from collections import defaultdict, Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import ray
from ray.air.execution._internal.event_manager import RayEventManager
from ray.air.execution.resources import (
from ray.air.execution._internal.tracked_actor import TrackedActor
from ray.air.execution._internal.tracked_actor_task import TrackedActorTask
from ray.exceptions import RayTaskError, RayActorError
def _enqueue_cached_actor_tasks(self, tracked_actor: TrackedActor):
    assert tracked_actor in self._live_actors_to_ray_actors_resources
    cached_tasks = self._pending_actors_to_enqueued_actor_tasks.pop(tracked_actor, [])
    for tracked_actor_task, method_name, args, kwargs in cached_tasks:
        self._schedule_tracked_actor_task(tracked_actor_task=tracked_actor_task, method_name=method_name, args=args, kwargs=kwargs)