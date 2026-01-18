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
def _cleanup_actor(self, tracked_actor: TrackedActor):
    self._cleanup_actor_futures(tracked_actor)
    ray_actor, acquired_resources = self._live_actors_to_ray_actors_resources.pop(tracked_actor)
    self._live_resource_cache = None
    self._resource_manager.free_resources(acquired_resource=acquired_resources)