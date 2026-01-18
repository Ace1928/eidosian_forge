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
class DeepChainMap(ChainMap):
    """Variant of ChainMap that allows direct updates to inner scopes.

        Taken from https://docs.python.org/3/library/collections.html#collections.ChainMap,
        with some light modifications for this use case.
        """

    def clear(self) -> None:
        for mapping in self.maps:
            mapping.clear()

    def __setitem__(self, key: KT, value: VT) -> None:
        for mapping in self.maps:
            mapping[key] = value

    def __delitem__(self, key: KT) -> None:
        hit = False
        for mapping in self.maps:
            if key in mapping:
                del mapping[key]
                hit = True
        if not hit:
            raise KeyError(key)