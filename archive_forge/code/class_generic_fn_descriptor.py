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
class generic_fn_descriptor(Generic[_T_co]):
    """Descriptor which proxies a function when the attribute is not
    present in dict

    This superclass is organized in a particular way with "memoized" and
    "non-memoized" implementation classes that are hidden from type checkers,
    as Mypy seems to not be able to handle seeing multiple kinds of descriptor
    classes used for the same attribute.

    """
    fget: Callable[..., _T_co]
    __doc__: Optional[str]
    __name__: str

    def __init__(self, fget: Callable[..., _T_co], doc: Optional[str]=None):
        self.fget = fget
        self.__doc__ = doc or fget.__doc__
        self.__name__ = fget.__name__

    @overload
    def __get__(self: _GFD, obj: None, cls: Any) -> _GFD:
        ...

    @overload
    def __get__(self, obj: object, cls: Any) -> _T_co:
        ...

    def __get__(self: _GFD, obj: Any, cls: Any) -> Union[_GFD, _T_co]:
        raise NotImplementedError()
    if TYPE_CHECKING:

        def __set__(self, instance: Any, value: Any) -> None:
            ...

        def __delete__(self, instance: Any) -> None:
            ...

    def _reset(self, obj: Any) -> None:
        raise NotImplementedError()

    @classmethod
    def reset(cls, obj: Any, name: str) -> None:
        raise NotImplementedError()