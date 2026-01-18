from __future__ import annotations as _annotations
import dataclasses
import keyword
import typing
import weakref
from collections import OrderedDict, defaultdict, deque
from copy import deepcopy
from itertools import zip_longest
from types import BuiltinFunctionType, CodeType, FunctionType, GeneratorType, LambdaType, ModuleType
from typing import Any, Mapping, TypeVar
from typing_extensions import TypeAlias, TypeGuard
from . import _repr, _typing_extra
@dataclasses.dataclass(frozen=True)
class SafeGetItemProxy:
    """Wrapper redirecting `__getitem__` to `get` with a sentinel value as default

    This makes is safe to use in `operator.itemgetter` when some keys may be missing
    """
    __slots__ = ('wrapped',)
    wrapped: Mapping[str, Any]

    def __getitem__(self, __key: str) -> Any:
        return self.wrapped.get(__key, _SENTINEL)
    if typing.TYPE_CHECKING:

        def __contains__(self, __key: str) -> bool:
            return self.wrapped.__contains__(__key)