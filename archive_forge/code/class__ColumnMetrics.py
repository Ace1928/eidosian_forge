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
class _ColumnMetrics(Generic[_COL_co]):
    __slots__ = ('column',)
    column: _COL_co

    def __init__(self, collection: ColumnCollection[Any, _COL_co], col: _COL_co):
        self.column = col
        pi = collection._proxy_index
        if pi:
            for eps_col in col._expanded_proxy_set:
                pi[eps_col].add(self)

    def get_expanded_proxy_set(self):
        return self.column._expanded_proxy_set

    def dispose(self, collection):
        pi = collection._proxy_index
        if not pi:
            return
        for col in self.column._expanded_proxy_set:
            colset = pi.get(col, None)
            if colset:
                colset.discard(self)
            if colset is not None and (not colset):
                del pi[col]

    def embedded(self, target_set: Union[Set[ColumnElement[Any]], FrozenSet[ColumnElement[Any]]]) -> bool:
        expanded_proxy_set = self.column._expanded_proxy_set
        for t in target_set.difference(expanded_proxy_set):
            if not expanded_proxy_set.intersection(_expand_cloned([t])):
                return False
        return True