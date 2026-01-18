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
class ClassAttribute:
    """Hide class attribute from its instances."""
    __slots__ = ('name', 'value')

    def __init__(self, name: str, value: Any) -> None:
        self.name = name
        self.value = value

    def __get__(self, instance: Any, owner: type[Any]) -> None:
        if instance is None:
            return self.value
        raise AttributeError(f'{self.name!r} attribute of {owner.__name__!r} is class-only')