import inspect
from contextlib import contextmanager
from typing import Any, Optional, Set, Tuple
import ray  # noqa: F401
import colorama
import ray.cloudpickle as cp
from ray.util.annotations import DeveloperAPI
def _inspect_generic_serialization(base_obj, depth, parent, failure_set, printer):
    """Adds the first-found non-serializable element to the failure_set."""
    assert not inspect.isfunction(base_obj)
    functions = inspect.getmembers(base_obj, predicate=inspect.isfunction)
    found = False
    with printer.indent():
        for name, obj in functions:
            serializable, _ = _inspect_serializability(obj, name=name, depth=depth - 1, parent=parent, failure_set=failure_set, printer=printer)
            found = found or not serializable
            if found:
                break
    with printer.indent():
        members = inspect.getmembers(base_obj)
        for name, obj in members:
            if name.startswith('__') and name.endswith('__') or inspect.isbuiltin(obj):
                continue
            serializable, _ = _inspect_serializability(obj, name=name, depth=depth - 1, parent=parent, failure_set=failure_set, printer=printer)
            found = found or not serializable
            if found:
                break
    if not found:
        printer.print(f'WARNING: Did not find non-serializable object in {base_obj}. This may be an oversight.')
    return found