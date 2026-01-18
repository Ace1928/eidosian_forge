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
class ColumnAdapter(ClauseAdapter):
    """Extends ClauseAdapter with extra utility functions.

    Key aspects of ColumnAdapter include:

    * Expressions that are adapted are stored in a persistent
      .columns collection; so that an expression E adapted into
      an expression E1, will return the same object E1 when adapted
      a second time.   This is important in particular for things like
      Label objects that are anonymized, so that the ColumnAdapter can
      be used to present a consistent "adapted" view of things.

    * Exclusion of items from the persistent collection based on
      include/exclude rules, but also independent of hash identity.
      This because "annotated" items all have the same hash identity as their
      parent.

    * "wrapping" capability is added, so that the replacement of an expression
      E can proceed through a series of adapters.  This differs from the
      visitor's "chaining" feature in that the resulting object is passed
      through all replacing functions unconditionally, rather than stopping
      at the first one that returns non-None.

    * An adapt_required option, used by eager loading to indicate that
      We don't trust a result row column that is not translated.
      This is to prevent a column from being interpreted as that
      of the child row in a self-referential scenario, see
      inheritance/test_basic.py->EagerTargetingTest.test_adapt_stringency

    """
    __slots__ = ('columns', 'adapt_required', 'allow_label_resolve', '_wrap', '__weakref__')
    columns: _ColumnLookup

    def __init__(self, selectable: Selectable, equivalents: Optional[_EquivalentColumnMap]=None, adapt_required: bool=False, include_fn: Optional[Callable[[ClauseElement], bool]]=None, exclude_fn: Optional[Callable[[ClauseElement], bool]]=None, adapt_on_names: bool=False, allow_label_resolve: bool=True, anonymize_labels: bool=False, adapt_from_selectables: Optional[AbstractSet[FromClause]]=None):
        super().__init__(selectable, equivalents, include_fn=include_fn, exclude_fn=exclude_fn, adapt_on_names=adapt_on_names, anonymize_labels=anonymize_labels, adapt_from_selectables=adapt_from_selectables)
        self.columns = util.WeakPopulateDict(self._locate_col)
        if self.include_fn or self.exclude_fn:
            self.columns = self._IncludeExcludeMapping(self, self.columns)
        self.adapt_required = adapt_required
        self.allow_label_resolve = allow_label_resolve
        self._wrap = None

    class _IncludeExcludeMapping:

        def __init__(self, parent, columns):
            self.parent = parent
            self.columns = columns

        def __getitem__(self, key):
            if self.parent.include_fn and (not self.parent.include_fn(key)) or (self.parent.exclude_fn and self.parent.exclude_fn(key)):
                if self.parent._wrap:
                    return self.parent._wrap.columns[key]
                else:
                    return key
            return self.columns[key]

    def wrap(self, adapter):
        ac = copy.copy(self)
        ac._wrap = adapter
        ac.columns = util.WeakPopulateDict(ac._locate_col)
        if ac.include_fn or ac.exclude_fn:
            ac.columns = self._IncludeExcludeMapping(ac, ac.columns)
        return ac

    @overload
    def traverse(self, obj: Literal[None]) -> None:
        ...

    @overload
    def traverse(self, obj: _ET) -> _ET:
        ...

    def traverse(self, obj: Optional[ExternallyTraversible]) -> Optional[ExternallyTraversible]:
        return self.columns[obj]

    def chain(self, visitor: ExternalTraversal) -> ColumnAdapter:
        assert isinstance(visitor, ColumnAdapter)
        return super().chain(visitor)
    if TYPE_CHECKING:

        @property
        def visitor_iterator(self) -> Iterator[ColumnAdapter]:
            ...
    adapt_clause = traverse
    adapt_list = ClauseAdapter.copy_and_process

    def adapt_check_present(self, col: ColumnElement[Any]) -> Optional[ColumnElement[Any]]:
        newcol = self.columns[col]
        if newcol is col and self._corresponding_column(col, True) is None:
            return None
        return newcol

    def _locate_col(self, col: ColumnElement[Any]) -> Optional[ColumnElement[Any]]:
        if col._is_immutable:
            for vis in self.visitor_iterator:
                c = vis.replace(col, _include_singleton_constants=True)
                if c is not None:
                    break
            else:
                c = col
        else:
            c = ClauseAdapter.traverse(self, col)
        if self._wrap:
            c2 = self._wrap._locate_col(c)
            if c2 is not None:
                c = c2
        if self.adapt_required and c is col:
            return None
        if c is not col:
            c._allow_label_resolve = self.allow_label_resolve
        return c