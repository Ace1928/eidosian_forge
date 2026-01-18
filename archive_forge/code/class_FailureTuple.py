import inspect
from contextlib import contextmanager
from typing import Any, Optional, Set, Tuple
import ray  # noqa: F401
import colorama
import ray.cloudpickle as cp
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
class FailureTuple:
    """Represents the serialization 'frame'.

    Attributes:
        obj: The object that fails serialization.
        name: The variable name of the object.
        parent: The object that references the `obj`.
    """

    def __init__(self, obj: Any, name: str, parent: Any):
        self.obj = obj
        self.name = name
        self.parent = parent

    def __repr__(self):
        return f'FailTuple({self.name} [obj={self.obj}, parent={self.parent}])'