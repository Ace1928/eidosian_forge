import inspect
import os
import sys
import warnings
from functools import partial
from importlib.metadata import entry_points
from ..exception import NetworkXNotImplemented
def _convert_and_call(self, backend_name, args, kwargs, *, fallback_to_nx=False):
    """Call this dispatchable function with a backend, converting graphs if necessary."""
    backend = _load_backend(backend_name)
    if not self._can_backend_run(backend_name, *args, **kwargs):
        if fallback_to_nx:
            return self.orig_func(*args, **kwargs)
        msg = f"'{self.name}' not implemented by {backend_name}"
        if hasattr(backend, self.name):
            msg += ' with the given arguments'
        raise RuntimeError(msg)
    try:
        converted_args, converted_kwargs = self._convert_arguments(backend_name, args, kwargs)
        result = getattr(backend, self.name)(*converted_args, **converted_kwargs)
    except (NotImplementedError, NetworkXNotImplemented) as exc:
        if fallback_to_nx:
            return self.orig_func(*args, **kwargs)
        raise
    return result