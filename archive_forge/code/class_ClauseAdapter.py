from __future__ import annotations
from collections import deque
import copy
from itertools import chain
import typing
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import operators
from . import roles
from . import visitors
from ._typing import is_text_clause
from .annotation import _deep_annotate as _deep_annotate  # noqa: F401
from .annotation import _deep_deannotate as _deep_deannotate  # noqa: F401
from .annotation import _shallow_annotate as _shallow_annotate  # noqa: F401
from .base import _expand_cloned
from .base import _from_objects
from .cache_key import HasCacheKey as HasCacheKey  # noqa: F401
from .ddl import sort_tables as sort_tables  # noqa: F401
from .elements import _find_columns as _find_columns
from .elements import _label_reference
from .elements import _textual_label_reference
from .elements import BindParameter
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import Grouping
from .elements import KeyedColumnElement
from .elements import Label
from .elements import NamedColumn
from .elements import Null
from .elements import UnaryExpression
from .schema import Column
from .selectable import Alias
from .selectable import FromClause
from .selectable import FromGrouping
from .selectable import Join
from .selectable import ScalarSelect
from .selectable import SelectBase
from .selectable import TableClause
from .visitors import _ET
from .. import exc
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
class ClauseAdapter(visitors.ReplacingExternalTraversal):
    """Clones and modifies clauses based on column correspondence.

    E.g.::

      table1 = Table('sometable', metadata,
          Column('col1', Integer),
          Column('col2', Integer)
          )
      table2 = Table('someothertable', metadata,
          Column('col1', Integer),
          Column('col2', Integer)
          )

      condition = table1.c.col1 == table2.c.col1

    make an alias of table1::

      s = table1.alias('foo')

    calling ``ClauseAdapter(s).traverse(condition)`` converts
    condition to read::

      s.c.col1 == table2.c.col1

    """
    __slots__ = ('__traverse_options__', 'selectable', 'include_fn', 'exclude_fn', 'equivalents', 'adapt_on_names', 'adapt_from_selectables')

    def __init__(self, selectable: Selectable, equivalents: Optional[_EquivalentColumnMap]=None, include_fn: Optional[Callable[[ClauseElement], bool]]=None, exclude_fn: Optional[Callable[[ClauseElement], bool]]=None, adapt_on_names: bool=False, anonymize_labels: bool=False, adapt_from_selectables: Optional[AbstractSet[FromClause]]=None):
        self.__traverse_options__ = {'stop_on': [selectable], 'anonymize_labels': anonymize_labels}
        self.selectable = selectable
        self.include_fn = include_fn
        self.exclude_fn = exclude_fn
        self.equivalents = util.column_dict(equivalents or {})
        self.adapt_on_names = adapt_on_names
        self.adapt_from_selectables = adapt_from_selectables
    if TYPE_CHECKING:

        @overload
        def traverse(self, obj: Literal[None]) -> None:
            ...

        @overload
        def traverse(self, obj: _ET) -> _ET:
            ...

        def traverse(self, obj: Optional[ExternallyTraversible]) -> Optional[ExternallyTraversible]:
            ...

    def _corresponding_column(self, col, require_embedded, _seen=util.EMPTY_SET):
        newcol = self.selectable.corresponding_column(col, require_embedded=require_embedded)
        if newcol is None and col in self.equivalents and (col not in _seen):
            for equiv in self.equivalents[col]:
                newcol = self._corresponding_column(equiv, require_embedded=require_embedded, _seen=_seen.union([col]))
                if newcol is not None:
                    return newcol
        if self.adapt_on_names and newcol is None and isinstance(col, NamedColumn):
            newcol = self.selectable.exported_columns.get(col.name)
        return newcol

    @util.preload_module('sqlalchemy.sql.functions')
    def replace(self, col: _ET, _include_singleton_constants: bool=False) -> Optional[_ET]:
        functions = util.preloaded.sql_functions
        if self.include_fn and (not self.include_fn(col)):
            return None
        elif self.exclude_fn and self.exclude_fn(col):
            return None
        if isinstance(col, FromClause) and (not isinstance(col, functions.FunctionElement)):
            if self.selectable.is_derived_from(col):
                if self.adapt_from_selectables:
                    for adp in self.adapt_from_selectables:
                        if adp.is_derived_from(col):
                            break
                    else:
                        return None
                return self.selectable
            elif isinstance(col, Alias) and isinstance(col.element, TableClause):
                return col
            else:
                return None
        elif not isinstance(col, ColumnElement):
            return None
        elif not _include_singleton_constants and col._is_singleton_constant:
            return None
        if 'adapt_column' in col._annotations:
            col = col._annotations['adapt_column']
        if TYPE_CHECKING:
            assert isinstance(col, KeyedColumnElement)
        if self.adapt_from_selectables and col not in self.equivalents:
            for adp in self.adapt_from_selectables:
                if adp.c.corresponding_column(col, False) is not None:
                    break
            else:
                return None
        if TYPE_CHECKING:
            assert isinstance(col, KeyedColumnElement)
        return self._corresponding_column(col, require_embedded=True)