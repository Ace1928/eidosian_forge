import inspect
import os
import sys
import warnings
from functools import partial
from importlib.metadata import entry_points
from ..exception import NetworkXNotImplemented
def _load_backend(backend_name):
    if backend_name in _loaded_backends:
        return _loaded_backends[backend_name]
    rv = _loaded_backends[backend_name] = backends[backend_name].load()
    return rv