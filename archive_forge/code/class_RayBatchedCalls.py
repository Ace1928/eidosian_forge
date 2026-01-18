import collections
import copy
import gc
import itertools
import logging
import os
import queue
import sys
import threading
import time
from multiprocessing import TimeoutError
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Tuple
import ray
from ray._private.usage import usage_lib
from ray.util import log_once
class RayBatchedCalls(BatchedCalls):
    """Joblib's BatchedCalls with basic Ray object store management

        This functionality is provided through the put_items_in_object_store,
        which uses external registries (list and dict) containing objects
        and their ObjectRefs."""

    def put_items_in_object_store(self, registry: Optional[List[Tuple[Any, ray.ObjectRef]]]=None, registry_hashable: Optional[Dict[Hashable, ray.ObjectRef]]=None):
        """Puts all applicable (kw)args in self.items in object store

            Takes two registries - list for unhashable objects and dict
            for hashable objects. The registries are a part of a Pool object.
            The method iterates through all entries in items list (usually,
            there will be only one, but the number depends on joblib Parallel
            settings) and puts all of the args and kwargs into the object
            store, updating the registries.
            If an arg or kwarg is already in a registry, it will not be
            put again, and instead, the cached object ref will be used."""
        new_items = []
        for func, args, kwargs in self.items:
            args = [ray_put_if_needed(arg, registry, registry_hashable) for arg in args]
            kwargs = {k: ray_put_if_needed(v, registry, registry_hashable) for k, v in kwargs.items()}
            new_items.append((func, args, kwargs))
        self.items = new_items

    def __call__(self):
        with parallel_backend(self._backend, n_jobs=self._n_jobs):
            return [func(*[ray_get_if_needed(arg) for arg in args], **{k: ray_get_if_needed(v) for k, v in kwargs.items()}) for func, args, kwargs in self.items]

    def __reduce__(self):
        if self._reducer_callback is not None:
            self._reducer_callback()
        return (RayBatchedCalls, (self.items, (self._backend, self._n_jobs), None, self._pickle_cache))