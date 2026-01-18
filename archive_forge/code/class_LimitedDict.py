from __future__ import annotations
import sys
import types
import typing
from collections import ChainMap
from contextlib import contextmanager
from contextvars import ContextVar
from types import prepare_class
from typing import TYPE_CHECKING, Any, Iterator, List, Mapping, MutableMapping, Tuple, TypeVar
from weakref import WeakValueDictionary
import typing_extensions
from ._core_utils import get_type_ref
from ._forward_ref import PydanticRecursiveRef
from ._typing_extra import TypeVarType, typing_base
from ._utils import all_identical, is_model_class
class LimitedDict(dict):
    """Limit the size/length of a dict used for caching to avoid unlimited increase in memory usage.

        Since the dict is ordered, and we always remove elements from the beginning, this is effectively a FIFO cache.
        """

    def __init__(self, size_limit: int=_LIMITED_DICT_SIZE):
        self.size_limit = size_limit
        super().__init__()

    def __setitem__(self, __key: Any, __value: Any) -> None:
        super().__setitem__(__key, __value)
        if len(self) > self.size_limit:
            excess = len(self) - self.size_limit + self.size_limit // 10
            to_remove = list(self.keys())[:excess]
            for key in to_remove:
                del self[key]