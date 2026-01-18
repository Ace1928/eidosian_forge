import weakref
from dataclasses import dataclass
import logging
from typing import List, TypeVar, Optional, Dict, Type, Tuple
import ray
from ray.actor import ActorHandle
from ray.util.annotations import Deprecated
from ray._private.utils import get_ray_doc_version
class ActorGroupMethod:

    def __init__(self, actor_group: 'ActorGroup', method_name: str):
        self.actor_group = weakref.ref(actor_group)
        self._method_name = method_name

    def __call__(self, *args, **kwargs):
        raise TypeError(f"ActorGroup methods cannot be called directly. Instead of running 'object.{self._method_name}()', try 'object.{self._method_name}.remote()'.")

    def remote(self, *args, **kwargs):
        return [getattr(a.actor, self._method_name).remote(*args, **kwargs) for a in self.actor_group().actors]