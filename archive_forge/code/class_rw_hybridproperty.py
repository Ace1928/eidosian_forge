from __future__ import annotations
import collections
import enum
from functools import update_wrapper
import inspect
import itertools
import operator
import re
import sys
import textwrap
import threading
import types
from types import CodeType
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import warnings
from . import _collections
from . import compat
from ._has_cy import HAS_CYEXTENSION
from .typing import Literal
from .. import exc
class rw_hybridproperty(Generic[_T]):

    def __init__(self, func: Callable[..., _T]):
        self.func = func
        self.clslevel = func
        self.setfn: Optional[Callable[..., Any]] = None

    def __get__(self, instance: Any, owner: Any) -> _T:
        if instance is None:
            clsval = self.clslevel(owner)
            return clsval
        else:
            return self.func(instance)

    def __set__(self, instance: Any, value: Any) -> None:
        assert self.setfn is not None
        self.setfn(instance, value)

    def setter(self, func: Callable[..., Any]) -> rw_hybridproperty[_T]:
        self.setfn = func
        return self

    def classlevel(self, func: Callable[..., Any]) -> rw_hybridproperty[_T]:
        self.clslevel = func
        return self