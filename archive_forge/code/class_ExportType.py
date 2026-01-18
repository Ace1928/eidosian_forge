from collections.abc import Sequence
import functools
import sys
from typing import Any, NamedTuple, Optional, Protocol, TypeVar
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
class ExportType(Protocol):

    def __call__(self, *v2: str, v1: Optional[Sequence[str]]=None, allow_multiple_exports: bool=True) -> api_export:
        ...