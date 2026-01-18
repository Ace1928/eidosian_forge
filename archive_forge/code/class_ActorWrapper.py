import abc
import functools
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import ray
from ray.actor import ActorHandle
from ray.air._internal.util import StartTraceback, find_free_port
from ray.exceptions import RayActorError
from ray.types import ObjectRef
class ActorWrapper:
    """Wraps an actor to provide same API as using the base class directly."""

    def __init__(self, actor: ActorHandle):
        self.actor = actor

    def __getattr__(self, item):
        actor_method = getattr(self.actor, item)
        return lambda *args, **kwargs: ray.get(actor_method.remote(*args, **kwargs))