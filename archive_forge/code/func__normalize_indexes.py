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
def _normalize_indexes(self, items: MappingIntStrAny, v_length: int) -> dict[int | str, Any]:
    """:param items: dict or set of indexes which will be normalized
        :param v_length: length of sequence indexes of which will be

        >>> self._normalize_indexes({0: True, -2: True, -1: True}, 4)
        {0: True, 2: True, 3: True}
        >>> self._normalize_indexes({'__all__': True}, 4)
        {0: True, 1: True, 2: True, 3: True}
        """
    normalized_items: dict[int | str, Any] = {}
    all_items = None
    for i, v in items.items():
        if not (isinstance(v, typing.Mapping) or isinstance(v, typing.AbstractSet) or self.is_true(v)):
            raise TypeError(f'Unexpected type of exclude value for index "{i}" {v.__class__}')
        if i == '__all__':
            all_items = self._coerce_value(v)
            continue
        if not isinstance(i, int):
            raise TypeError('Excluding fields from a sequence of sub-models or dicts must be performed index-wise: expected integer keys or keyword "__all__"')
        normalized_i = v_length + i if i < 0 else i
        normalized_items[normalized_i] = self.merge(v, normalized_items.get(normalized_i))
    if not all_items:
        return normalized_items
    if self.is_true(all_items):
        for i in range(v_length):
            normalized_items.setdefault(i, ...)
        return normalized_items
    for i in range(v_length):
        normalized_item = normalized_items.setdefault(i, {})
        if not self.is_true(normalized_item):
            normalized_items[i] = self.merge(all_items, normalized_item)
    return normalized_items