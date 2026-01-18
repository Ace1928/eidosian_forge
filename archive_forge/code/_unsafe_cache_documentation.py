import functools
from typing import Any, Callable, Dict, List, TypeVar
Cache decorator that relies object IDs when arguments are unhashable. Makes the
    very strong assumption of not only immutability, but that unhashable types don't go
    out of scope.