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
def add_actor(self, cls: Union[Type, ray.actor.ActorClass], kwargs: Dict[str, Any], resource_request: ResourceRequest, *, on_start: Optional[Callable[[TrackedActor], None]]=None, on_stop: Optional[Callable[[TrackedActor], None]]=None, on_error: Optional[Callable[[TrackedActor, Exception], None]]=None) -> TrackedActor:
    """Add an actor to be tracked.

        This method will request resources to start the actor. Once the resources
        are available, the actor will be started and the
        :meth:`TrackedActor.on_start
        <ray.air.execution._internal.tracked_actor.TrackedActor.on_start>` callback
        will be invoked.

        Args:
            cls: Actor class to schedule.
            kwargs: Keyword arguments to pass to actor class on construction.
            resource_request: Resources required to start the actor.
            on_start: Callback to invoke when the actor started.
            on_stop: Callback to invoke when the actor stopped.
            on_error: Callback to invoke when the actor failed.

        Returns:
            Tracked actor object to reference actor in subsequent API calls.

        """
    tracked_actor = TrackedActor(uuid.uuid4().int, on_start=on_start, on_stop=on_stop, on_error=on_error)
    self._pending_actors_to_attrs[tracked_actor] = (cls, kwargs, resource_request)
    self._resource_request_to_pending_actors[resource_request].append(tracked_actor)
    self._resource_manager.request_resources(resource_request=resource_request)
    return tracked_actor