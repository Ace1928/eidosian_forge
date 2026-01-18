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
class ColumnCollection(Generic[_COLKEY, _COL_co]):
    """Collection of :class:`_expression.ColumnElement` instances,
    typically for
    :class:`_sql.FromClause` objects.

    The :class:`_sql.ColumnCollection` object is most commonly available
    as the :attr:`_schema.Table.c` or :attr:`_schema.Table.columns` collection
    on the :class:`_schema.Table` object, introduced at
    :ref:`metadata_tables_and_columns`.

    The :class:`_expression.ColumnCollection` has both mapping- and sequence-
    like behaviors. A :class:`_expression.ColumnCollection` usually stores
    :class:`_schema.Column` objects, which are then accessible both via mapping
    style access as well as attribute access style.

    To access :class:`_schema.Column` objects using ordinary attribute-style
    access, specify the name like any other object attribute, such as below
    a column named ``employee_name`` is accessed::

        >>> employee_table.c.employee_name

    To access columns that have names with special characters or spaces,
    index-style access is used, such as below which illustrates a column named
    ``employee ' payment`` is accessed::

        >>> employee_table.c["employee ' payment"]

    As the :class:`_sql.ColumnCollection` object provides a Python dictionary
    interface, common dictionary method names like
    :meth:`_sql.ColumnCollection.keys`, :meth:`_sql.ColumnCollection.values`,
    and :meth:`_sql.ColumnCollection.items` are available, which means that
    database columns that are keyed under these names also need to use indexed
    access::

        >>> employee_table.c["values"]


    The name for which a :class:`_schema.Column` would be present is normally
    that of the :paramref:`_schema.Column.key` parameter.  In some contexts,
    such as a :class:`_sql.Select` object that uses a label style set
    using the :meth:`_sql.Select.set_label_style` method, a column of a certain
    key may instead be represented under a particular label name such
    as ``tablename_columnname``::

        >>> from sqlalchemy import select, column, table
        >>> from sqlalchemy import LABEL_STYLE_TABLENAME_PLUS_COL
        >>> t = table("t", column("c"))
        >>> stmt = select(t).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)
        >>> subq = stmt.subquery()
        >>> subq.c.t_c
        <sqlalchemy.sql.elements.ColumnClause at 0x7f59dcf04fa0; t_c>

    :class:`.ColumnCollection` also indexes the columns in order and allows
    them to be accessible by their integer position::

        >>> cc[0]
        Column('x', Integer(), table=None)
        >>> cc[1]
        Column('y', Integer(), table=None)

    .. versionadded:: 1.4 :class:`_expression.ColumnCollection`
       allows integer-based
       index access to the collection.

    Iterating the collection yields the column expressions in order::

        >>> list(cc)
        [Column('x', Integer(), table=None),
         Column('y', Integer(), table=None)]

    The base :class:`_expression.ColumnCollection` object can store
    duplicates, which can
    mean either two columns with the same key, in which case the column
    returned by key  access is **arbitrary**::

        >>> x1, x2 = Column('x', Integer), Column('x', Integer)
        >>> cc = ColumnCollection(columns=[(x1.name, x1), (x2.name, x2)])
        >>> list(cc)
        [Column('x', Integer(), table=None),
         Column('x', Integer(), table=None)]
        >>> cc['x'] is x1
        False
        >>> cc['x'] is x2
        True

    Or it can also mean the same column multiple times.   These cases are
    supported as :class:`_expression.ColumnCollection`
    is used to represent the columns in
    a SELECT statement which may include duplicates.

    A special subclass :class:`.DedupeColumnCollection` exists which instead
    maintains SQLAlchemy's older behavior of not allowing duplicates; this
    collection is used for schema level objects like :class:`_schema.Table`
    and
    :class:`.PrimaryKeyConstraint` where this deduping is helpful.  The
    :class:`.DedupeColumnCollection` class also has additional mutation methods
    as the schema constructs have more use cases that require removal and
    replacement of columns.

    .. versionchanged:: 1.4 :class:`_expression.ColumnCollection`
       now stores duplicate
       column keys as well as the same column in multiple positions.  The
       :class:`.DedupeColumnCollection` class is added to maintain the
       former behavior in those cases where deduplication as well as
       additional replace/remove operations are needed.


    """
    __slots__ = ('_collection', '_index', '_colset', '_proxy_index')
    _collection: List[Tuple[_COLKEY, _COL_co, _ColumnMetrics[_COL_co]]]
    _index: Dict[Union[None, str, int], Tuple[_COLKEY, _COL_co]]
    _proxy_index: Dict[ColumnElement[Any], Set[_ColumnMetrics[_COL_co]]]
    _colset: Set[_COL_co]

    def __init__(self, columns: Optional[Iterable[Tuple[_COLKEY, _COL_co]]]=None):
        object.__setattr__(self, '_colset', set())
        object.__setattr__(self, '_index', {})
        object.__setattr__(self, '_proxy_index', collections.defaultdict(util.OrderedSet))
        object.__setattr__(self, '_collection', [])
        if columns:
            self._initial_populate(columns)

    @util.preload_module('sqlalchemy.sql.elements')
    def __clause_element__(self) -> ClauseList:
        elements = util.preloaded.sql_elements
        return elements.ClauseList(*self._all_columns, _literal_as_text_role=roles.ColumnsClauseRole, group=False)

    def _initial_populate(self, iter_: Iterable[Tuple[_COLKEY, _COL_co]]) -> None:
        self._populate_separate_keys(iter_)

    @property
    def _all_columns(self) -> List[_COL_co]:
        return [col for _, col, _ in self._collection]

    def keys(self) -> List[_COLKEY]:
        """Return a sequence of string key names for all columns in this
        collection."""
        return [k for k, _, _ in self._collection]

    def values(self) -> List[_COL_co]:
        """Return a sequence of :class:`_sql.ColumnClause` or
        :class:`_schema.Column` objects for all columns in this
        collection."""
        return [col for _, col, _ in self._collection]

    def items(self) -> List[Tuple[_COLKEY, _COL_co]]:
        """Return a sequence of (key, column) tuples for all columns in this
        collection each consisting of a string key name and a
        :class:`_sql.ColumnClause` or
        :class:`_schema.Column` object.
        """
        return [(k, col) for k, col, _ in self._collection]

    def __bool__(self) -> bool:
        return bool(self._collection)

    def __len__(self) -> int:
        return len(self._collection)

    def __iter__(self) -> Iterator[_COL_co]:
        return iter([col for _, col, _ in self._collection])

    @overload
    def __getitem__(self, key: Union[str, int]) -> _COL_co:
        ...

    @overload
    def __getitem__(self, key: Tuple[Union[str, int], ...]) -> ReadOnlyColumnCollection[_COLKEY, _COL_co]:
        ...

    @overload
    def __getitem__(self, key: slice) -> ReadOnlyColumnCollection[_COLKEY, _COL_co]:
        ...

    def __getitem__(self, key: Union[str, int, slice, Tuple[Union[str, int], ...]]) -> Union[ReadOnlyColumnCollection[_COLKEY, _COL_co], _COL_co]:
        try:
            if isinstance(key, (tuple, slice)):
                if isinstance(key, slice):
                    cols = ((sub_key, col) for sub_key, col, _ in self._collection[key])
                else:
                    cols = (self._index[sub_key] for sub_key in key)
                return ColumnCollection(cols).as_readonly()
            else:
                return self._index[key][1]
        except KeyError as err:
            if isinstance(err.args[0], int):
                raise IndexError(err.args[0]) from err
            else:
                raise

    def __getattr__(self, key: str) -> _COL_co:
        try:
            return self._index[key][1]
        except KeyError as err:
            raise AttributeError(key) from err

    def __contains__(self, key: str) -> bool:
        if key not in self._index:
            if not isinstance(key, str):
                raise exc.ArgumentError('__contains__ requires a string argument')
            return False
        else:
            return True

    def compare(self, other: ColumnCollection[Any, Any]) -> bool:
        """Compare this :class:`_expression.ColumnCollection` to another
        based on the names of the keys"""
        for l, r in zip_longest(self, other):
            if l is not r:
                return False
        else:
            return True

    def __eq__(self, other: Any) -> bool:
        return self.compare(other)

    def get(self, key: str, default: Optional[_COL_co]=None) -> Optional[_COL_co]:
        """Get a :class:`_sql.ColumnClause` or :class:`_schema.Column` object
        based on a string key name from this
        :class:`_expression.ColumnCollection`."""
        if key in self._index:
            return self._index[key][1]
        else:
            return default

    def __str__(self) -> str:
        return '%s(%s)' % (self.__class__.__name__, ', '.join((str(c) for c in self)))

    def __setitem__(self, key: str, value: Any) -> NoReturn:
        raise NotImplementedError()

    def __delitem__(self, key: str) -> NoReturn:
        raise NotImplementedError()

    def __setattr__(self, key: str, obj: Any) -> NoReturn:
        raise NotImplementedError()

    def clear(self) -> NoReturn:
        """Dictionary clear() is not implemented for
        :class:`_sql.ColumnCollection`."""
        raise NotImplementedError()

    def remove(self, column: Any) -> None:
        raise NotImplementedError()

    def update(self, iter_: Any) -> NoReturn:
        """Dictionary update() is not implemented for
        :class:`_sql.ColumnCollection`."""
        raise NotImplementedError()
    __hash__ = None

    def _populate_separate_keys(self, iter_: Iterable[Tuple[_COLKEY, _COL_co]]) -> None:
        """populate from an iterator of (key, column)"""
        self._collection[:] = collection = [(k, c, _ColumnMetrics(self, c)) for k, c in iter_]
        self._colset.update((c._deannotate() for _, c, _ in collection))
        self._index.update({idx: (k, c) for idx, (k, c, _) in enumerate(collection)})
        self._index.update({k: (k, col) for k, col, _ in reversed(collection)})

    def add(self, column: ColumnElement[Any], key: Optional[_COLKEY]=None) -> None:
        """Add a column to this :class:`_sql.ColumnCollection`.

        .. note::

            This method is **not normally used by user-facing code**, as the
            :class:`_sql.ColumnCollection` is usually part of an existing
            object such as a :class:`_schema.Table`. To add a
            :class:`_schema.Column` to an existing :class:`_schema.Table`
            object, use the :meth:`_schema.Table.append_column` method.

        """
        colkey: _COLKEY
        if key is None:
            colkey = column.key
        else:
            colkey = key
        l = len(self._collection)
        _column = cast(_COL_co, column)
        self._collection.append((colkey, _column, _ColumnMetrics(self, _column)))
        self._colset.add(_column._deannotate())
        self._index[l] = (colkey, _column)
        if colkey not in self._index:
            self._index[colkey] = (colkey, _column)

    def __getstate__(self) -> Dict[str, Any]:
        return {'_collection': [(k, c) for k, c, _ in self._collection], '_index': self._index}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        object.__setattr__(self, '_index', state['_index'])
        object.__setattr__(self, '_proxy_index', collections.defaultdict(util.OrderedSet))
        object.__setattr__(self, '_collection', [(k, c, _ColumnMetrics(self, c)) for k, c in state['_collection']])
        object.__setattr__(self, '_colset', {col for k, col, _ in self._collection})

    def contains_column(self, col: ColumnElement[Any]) -> bool:
        """Checks if a column object exists in this collection"""
        if col not in self._colset:
            if isinstance(col, str):
                raise exc.ArgumentError('contains_column cannot be used with string arguments. Use ``col_name in table.c`` instead.')
            return False
        else:
            return True

    def as_readonly(self) -> ReadOnlyColumnCollection[_COLKEY, _COL_co]:
        """Return a "read only" form of this
        :class:`_sql.ColumnCollection`."""
        return ReadOnlyColumnCollection(self)

    def _init_proxy_index(self):
        """populate the "proxy index", if empty.

        proxy index is added in 2.0 to provide more efficient operation
        for the corresponding_column() method.

        For reasons of both time to construct new .c collections as well as
        memory conservation for large numbers of large .c collections, the
        proxy_index is only filled if corresponding_column() is called. once
        filled it stays that way, and new _ColumnMetrics objects created after
        that point will populate it with new data. Note this case would be
        unusual, if not nonexistent, as it means a .c collection is being
        mutated after corresponding_column() were used, however it is tested in
        test/base/test_utils.py.

        """
        pi = self._proxy_index
        if pi:
            return
        for _, _, metrics in self._collection:
            eps = metrics.column._expanded_proxy_set
            for eps_col in eps:
                pi[eps_col].add(metrics)

    def corresponding_column(self, column: _COL, require_embedded: bool=False) -> Optional[Union[_COL, _COL_co]]:
        """Given a :class:`_expression.ColumnElement`, return the exported
        :class:`_expression.ColumnElement` object from this
        :class:`_expression.ColumnCollection`
        which corresponds to that original :class:`_expression.ColumnElement`
        via a common
        ancestor column.

        :param column: the target :class:`_expression.ColumnElement`
                      to be matched.

        :param require_embedded: only return corresponding columns for
         the given :class:`_expression.ColumnElement`, if the given
         :class:`_expression.ColumnElement`
         is actually present within a sub-element
         of this :class:`_expression.Selectable`.
         Normally the column will match if
         it merely shares a common ancestor with one of the exported
         columns of this :class:`_expression.Selectable`.

        .. seealso::

            :meth:`_expression.Selectable.corresponding_column`
            - invokes this method
            against the collection returned by
            :attr:`_expression.Selectable.exported_columns`.

        .. versionchanged:: 1.4 the implementation for ``corresponding_column``
           was moved onto the :class:`_expression.ColumnCollection` itself.

        """
        if column in self._colset:
            return column
        selected_intersection, selected_metrics = (None, None)
        target_set = column.proxy_set
        pi = self._proxy_index
        if not pi:
            self._init_proxy_index()
        for current_metrics in (mm for ts in target_set if ts in pi for mm in pi[ts]):
            if not require_embedded or current_metrics.embedded(target_set):
                if selected_metrics is None:
                    selected_metrics = current_metrics
                    continue
                current_intersection = target_set.intersection(current_metrics.column._expanded_proxy_set)
                if selected_intersection is None:
                    selected_intersection = target_set.intersection(selected_metrics.column._expanded_proxy_set)
                if len(current_intersection) > len(selected_intersection):
                    selected_metrics = current_metrics
                    selected_intersection = current_intersection
                elif current_intersection == selected_intersection:
                    selected_col_distance = sum([sc._annotations.get('weight', 1) for sc in selected_metrics.column._uncached_proxy_list() if sc.shares_lineage(column)])
                    current_col_distance = sum([sc._annotations.get('weight', 1) for sc in current_metrics.column._uncached_proxy_list() if sc.shares_lineage(column)])
                    if current_col_distance < selected_col_distance:
                        selected_metrics = current_metrics
                        selected_intersection = current_intersection
        return selected_metrics.column if selected_metrics else None