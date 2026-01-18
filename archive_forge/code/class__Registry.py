import atexit
import logging
from functools import partial
from types import FunctionType
from typing import Callable, Optional, Type, Union
import ray
import ray.cloudpickle as pickle
from ray.experimental.internal_kv import (
from ray.tune.error import TuneError
from ray.util.annotations import DeveloperAPI
class _Registry:

    def __init__(self, prefix: Optional[str]=None):
        """If no prefix is given, use runtime context job ID."""
        self._to_flush = {}
        self._prefix = prefix
        self._registered = set()
        self._atexit_handler_registered = False

    @property
    def prefix(self):
        if not self._prefix:
            self._prefix = ray.get_runtime_context().get_job_id()
        return self._prefix

    def _register_atexit(self):
        if self._atexit_handler_registered:
            return
        if ray._private.worker.global_worker.mode != ray.SCRIPT_MODE:
            return
        atexit.register(_unregister_all)
        self._atexit_handler_registered = True

    def register(self, category, key, value):
        """Registers the value with the global registry.

        Raises:
            PicklingError if unable to pickle to provided file.
        """
        if category not in KNOWN_CATEGORIES:
            from ray.tune import TuneError
            raise TuneError('Unknown category {} not among {}'.format(category, KNOWN_CATEGORIES))
        self._to_flush[category, key] = pickle.dumps_debug(value)
        if _internal_kv_initialized():
            self.flush_values()

    def unregister(self, category, key):
        if _internal_kv_initialized():
            _internal_kv_del(_make_key(self.prefix, category, key))
        else:
            self._to_flush.pop((category, key), None)

    def unregister_all(self, category: Optional[str]=None):
        remaining = set()
        for cat, key in self._registered:
            if category and category == cat:
                self.unregister(cat, key)
            else:
                remaining.add((cat, key))
        self._registered = remaining

    def contains(self, category, key):
        if _internal_kv_initialized():
            value = _internal_kv_get(_make_key(self.prefix, category, key))
            return value is not None
        else:
            return (category, key) in self._to_flush

    def get(self, category, key):
        if _internal_kv_initialized():
            value = _internal_kv_get(_make_key(self.prefix, category, key))
            if value is None:
                raise ValueError("Registry value for {}/{} doesn't exist.".format(category, key))
            return pickle.loads(value)
        else:
            return pickle.loads(self._to_flush[category, key])

    def flush_values(self):
        self._register_atexit()
        for (category, key), value in self._to_flush.items():
            _internal_kv_put(_make_key(self.prefix, category, key), value, overwrite=True)
            self._registered.add((category, key))
        self._to_flush.clear()