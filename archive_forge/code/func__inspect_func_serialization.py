import inspect
from contextlib import contextmanager
from typing import Any, Optional, Set, Tuple
import ray  # noqa: F401
import colorama
import ray.cloudpickle as cp
from ray.util.annotations import DeveloperAPI
def _inspect_func_serialization(base_obj, depth, parent, failure_set, printer):
    """Adds the first-found non-serializable element to the failure_set."""
    assert inspect.isfunction(base_obj)
    closure = inspect.getclosurevars(base_obj)
    found = False
    if closure.globals:
        printer.print(f'Detected {len(closure.globals)} global variables. Checking serializability...')
        with printer.indent():
            for name, obj in closure.globals.items():
                serializable, _ = _inspect_serializability(obj, name=name, depth=depth - 1, parent=parent, failure_set=failure_set, printer=printer)
                found = found or not serializable
                if found:
                    break
    if closure.nonlocals:
        printer.print(f'Detected {len(closure.nonlocals)} nonlocal variables. Checking serializability...')
        with printer.indent():
            for name, obj in closure.nonlocals.items():
                serializable, _ = _inspect_serializability(obj, name=name, depth=depth - 1, parent=parent, failure_set=failure_set, printer=printer)
                found = found or not serializable
                if found:
                    break
    if not found:
        printer.print(f'WARNING: Did not find non-serializable object in {base_obj}. This may be an oversight.')
    return found