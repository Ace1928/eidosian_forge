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
def _try_start_actors(self, max_actors: Optional[int]=None) -> int:
    """Try to start up to ``max_actors`` actors.

        This function will iterate through all resource requests we collected for
        pending actors. As long as a resource request can be fulfilled (resources
        are available), we try to start as many actors as possible.

        This will schedule a `Actor.__ray_ready__()` future which, once resolved,
        will trigger the `TrackedActor.on_start` callback.
        """
    started_actors = 0
    for resource_request in self._resource_request_to_pending_actors:
        if max_actors and started_actors >= max_actors:
            break
        while self._resource_manager.has_resources_ready(resource_request) and self._resource_request_to_pending_actors[resource_request]:
            acquired_resources = self._resource_manager.acquire_resources(resource_request)
            assert acquired_resources
            candidate_actors = self._resource_request_to_pending_actors[resource_request]
            assert candidate_actors
            tracked_actor = candidate_actors.pop(0)
            actor_cls, kwargs, _ = self._pending_actors_to_attrs.pop(tracked_actor)
            if not isinstance(actor_cls, ray.actor.ActorClass):
                actor_cls = ray.remote(actor_cls)
            [remote_actor_cls] = acquired_resources.annotate_remote_entities([actor_cls])
            actor = remote_actor_cls.remote(**kwargs)
            self._live_actors_to_ray_actors_resources[tracked_actor] = (actor, acquired_resources)
            self._live_resource_cache = None
            future = actor.__ray_ready__.remote()
            self._tracked_actors_to_state_futures[tracked_actor].add(future)

            def create_callbacks(tracked_actor: TrackedActor, future: ray.ObjectRef):

                def on_actor_start(result: Any):
                    self._actor_start_resolved(tracked_actor=tracked_actor, future=future)

                def on_error(exception: Exception):
                    self._actor_start_failed(tracked_actor=tracked_actor, exception=exception)
                return (on_actor_start, on_error)
            on_actor_start, on_error = create_callbacks(tracked_actor=tracked_actor, future=future)
            self._actor_state_events.track_future(future=future, on_result=on_actor_start, on_error=on_error)
            self._enqueue_cached_actor_tasks(tracked_actor=tracked_actor)
    return started_actors