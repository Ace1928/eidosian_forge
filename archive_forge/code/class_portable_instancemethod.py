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
class portable_instancemethod:
    """Turn an instancemethod into a (parent, name) pair
    to produce a serializable callable.

    """
    __slots__ = ('target', 'name', 'kwargs', '__weakref__')

    def __getstate__(self):
        return {'target': self.target, 'name': self.name, 'kwargs': self.kwargs}

    def __setstate__(self, state):
        self.target = state['target']
        self.name = state['name']
        self.kwargs = state.get('kwargs', ())

    def __init__(self, meth, kwargs=()):
        self.target = meth.__self__
        self.name = meth.__name__
        self.kwargs = kwargs

    def __call__(self, *arg, **kw):
        kw.update(self.kwargs)
        return getattr(self.target, self.name)(*arg, **kw)