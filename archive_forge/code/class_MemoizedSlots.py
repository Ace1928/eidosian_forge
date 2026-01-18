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
class MemoizedSlots:
    """Apply memoized items to an object using a __getattr__ scheme.

    This allows the functionality of memoized_property and
    memoized_instancemethod to be available to a class using __slots__.

    """
    __slots__ = ()

    def _fallback_getattr(self, key):
        raise AttributeError(key)

    def __getattr__(self, key: str) -> Any:
        if key.startswith('_memoized_attr_') or key.startswith('_memoized_method_'):
            raise AttributeError(key)
        elif hasattr(self.__class__, f'_memoized_attr_{key}'):
            value = getattr(self, f'_memoized_attr_{key}')()
            setattr(self, key, value)
            return value
        elif hasattr(self.__class__, f'_memoized_method_{key}'):
            fn = getattr(self, f'_memoized_method_{key}')

            def oneshot(*args, **kw):
                result = fn(*args, **kw)

                def memo(*a, **kw):
                    return result
                memo.__name__ = fn.__name__
                memo.__doc__ = fn.__doc__
                setattr(self, key, memo)
                return result
            oneshot.__doc__ = fn.__doc__
            return oneshot
        else:
            return self._fallback_getattr(key)