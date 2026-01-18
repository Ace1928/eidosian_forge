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
class hybridmethod(Generic[_T]):
    """Decorate a function as cls- or instance- level."""

    def __init__(self, func: Callable[..., _T]):
        self.func = self.__func__ = func
        self.clslevel = func

    def __get__(self, instance: Any, owner: Any) -> Callable[..., _T]:
        if instance is None:
            return self.clslevel.__get__(owner, owner.__class__)
        else:
            return self.func.__get__(instance, owner)

    def classlevel(self, func: Callable[..., Any]) -> hybridmethod[_T]:
        self.clslevel = func
        return self