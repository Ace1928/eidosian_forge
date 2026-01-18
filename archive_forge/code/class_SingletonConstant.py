from __future__ import annotations
import collections
from enum import Enum
import itertools
from itertools import zip_longest
import operator
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NamedTuple
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
from . import roles
from . import visitors
from .cache_key import HasCacheKey  # noqa
from .cache_key import MemoizedHasCacheKey  # noqa
from .traversals import HasCopyInternals  # noqa
from .visitors import ClauseVisitor
from .visitors import ExtendedInternalTraversal
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .. import event
from .. import exc
from .. import util
from ..util import HasMemoized as HasMemoized
from ..util import hybridmethod
from ..util import typing as compat_typing
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypeGuard
class SingletonConstant(Immutable):
    """Represent SQL constants like NULL, TRUE, FALSE"""
    _is_singleton_constant = True
    _singleton: SingletonConstant

    def __new__(cls: _T, *arg: Any, **kw: Any) -> _T:
        return cast(_T, cls._singleton)

    @util.non_memoized_property
    def proxy_set(self) -> FrozenSet[ColumnElement[Any]]:
        raise NotImplementedError()

    @classmethod
    def _create_singleton(cls):
        obj = object.__new__(cls)
        obj.__init__()
        obj.proxy_set = frozenset([obj])
        cls._singleton = obj