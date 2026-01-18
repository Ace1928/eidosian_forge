import os
from typing import Any, TypeVar, Callable, Optional, cast
from typing import Protocol
class _FunctionWithPrevious(Protocol[F]):
    previous: Optional[int]
    __call__: F