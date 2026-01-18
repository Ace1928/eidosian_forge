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
class DedupeColumnCollection(ColumnCollection[str, _NAMEDCOL]):
    """A :class:`_expression.ColumnCollection`
    that maintains deduplicating behavior.

    This is useful by schema level objects such as :class:`_schema.Table` and
    :class:`.PrimaryKeyConstraint`.    The collection includes more
    sophisticated mutator methods as well to suit schema objects which
    require mutable column collections.

    .. versionadded:: 1.4

    """

    def add(self, column: ColumnElement[Any], key: Optional[str]=None) -> None:
        named_column = cast(_NAMEDCOL, column)
        if key is not None and named_column.key != key:
            raise exc.ArgumentError('DedupeColumnCollection requires columns be under the same key as their .key')
        key = named_column.key
        if key is None:
            raise exc.ArgumentError("Can't add unnamed column to column collection")
        if key in self._index:
            existing = self._index[key][1]
            if existing is named_column:
                return
            self.replace(named_column)
            util.memoized_property.reset(named_column, 'proxy_set')
        else:
            self._append_new_column(key, named_column)

    def _append_new_column(self, key: str, named_column: _NAMEDCOL) -> None:
        l = len(self._collection)
        self._collection.append((key, named_column, _ColumnMetrics(self, named_column)))
        self._colset.add(named_column._deannotate())
        self._index[l] = (key, named_column)
        self._index[key] = (key, named_column)

    def _populate_separate_keys(self, iter_: Iterable[Tuple[str, _NAMEDCOL]]) -> None:
        """populate from an iterator of (key, column)"""
        cols = list(iter_)
        replace_col = []
        for k, col in cols:
            if col.key != k:
                raise exc.ArgumentError('DedupeColumnCollection requires columns be under the same key as their .key')
            if col.name in self._index and col.key != col.name:
                replace_col.append(col)
            elif col.key in self._index:
                replace_col.append(col)
            else:
                self._index[k] = (k, col)
                self._collection.append((k, col, _ColumnMetrics(self, col)))
        self._colset.update((c._deannotate() for k, c, _ in self._collection))
        self._index.update(((idx, (k, c)) for idx, (k, c, _) in enumerate(self._collection)))
        for col in replace_col:
            self.replace(col)

    def extend(self, iter_: Iterable[_NAMEDCOL]) -> None:
        self._populate_separate_keys(((col.key, col) for col in iter_))

    def remove(self, column: _NAMEDCOL) -> None:
        if column not in self._colset:
            raise ValueError("Can't remove column %r; column is not in this collection" % column)
        del self._index[column.key]
        self._colset.remove(column)
        self._collection[:] = [(k, c, metrics) for k, c, metrics in self._collection if c is not column]
        for metrics in self._proxy_index.get(column, ()):
            metrics.dispose(self)
        self._index.update({idx: (k, col) for idx, (k, col, _) in enumerate(self._collection)})
        del self._index[len(self._collection)]

    def replace(self, column: _NAMEDCOL, extra_remove: Optional[Iterable[_NAMEDCOL]]=None) -> None:
        """add the given column to this collection, removing unaliased
        versions of this column  as well as existing columns with the
        same key.

        e.g.::

            t = Table('sometable', metadata, Column('col1', Integer))
            t.columns.replace(Column('col1', Integer, key='columnone'))

        will remove the original 'col1' from the collection, and add
        the new column under the name 'columnname'.

        Used by schema.Column to override columns during table reflection.

        """
        if extra_remove:
            remove_col = set(extra_remove)
        else:
            remove_col = set()
        if column.name in self._index and column.key != column.name:
            other = self._index[column.name][1]
            if other.name == other.key:
                remove_col.add(other)
        if column.key in self._index:
            remove_col.add(self._index[column.key][1])
        if not remove_col:
            self._append_new_column(column.key, column)
            return
        new_cols: List[Tuple[str, _NAMEDCOL, _ColumnMetrics[_NAMEDCOL]]] = []
        replaced = False
        for k, col, metrics in self._collection:
            if col in remove_col:
                if not replaced:
                    replaced = True
                    new_cols.append((column.key, column, _ColumnMetrics(self, column)))
            else:
                new_cols.append((k, col, metrics))
        if remove_col:
            self._colset.difference_update(remove_col)
            for rc in remove_col:
                for metrics in self._proxy_index.get(rc, ()):
                    metrics.dispose(self)
        if not replaced:
            new_cols.append((column.key, column, _ColumnMetrics(self, column)))
        self._colset.add(column._deannotate())
        self._collection[:] = new_cols
        self._index.clear()
        self._index.update({idx: (k, col) for idx, (k, col, _) in enumerate(self._collection)})
        self._index.update({k: (k, col) for k, col, _ in self._collection})