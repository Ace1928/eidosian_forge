from __future__ import annotations
import collections
from enum import Enum
import itertools
from typing import AbstractSet
from typing import Any as TODO_Any
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
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
from . import cache_key
from . import coercions
from . import operators
from . import roles
from . import traversals
from . import type_api
from . import visitors
from ._typing import _ColumnsClauseArgument
from ._typing import _no_kw
from ._typing import _TP
from ._typing import is_column_element
from ._typing import is_select_statement
from ._typing import is_subquery
from ._typing import is_table
from ._typing import is_text_clause
from .annotation import Annotated
from .annotation import SupportsCloneAnnotations
from .base import _clone
from .base import _cloned_difference
from .base import _cloned_intersection
from .base import _entity_namespace_key
from .base import _EntityNamespace
from .base import _expand_cloned
from .base import _from_objects
from .base import _generative
from .base import _never_select_column
from .base import _NoArg
from .base import _select_iterables
from .base import CacheableOptions
from .base import ColumnCollection
from .base import ColumnSet
from .base import CompileState
from .base import DedupeColumnCollection
from .base import Executable
from .base import Generative
from .base import HasCompileState
from .base import HasMemoized
from .base import Immutable
from .coercions import _document_text_coercion
from .elements import _anonymous_label
from .elements import BindParameter
from .elements import BooleanClauseList
from .elements import ClauseElement
from .elements import ClauseList
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import DQLDMLClauseElement
from .elements import GroupedElement
from .elements import literal_column
from .elements import TableValuedColumn
from .elements import UnaryExpression
from .operators import OperatorType
from .sqltypes import NULLTYPE
from .visitors import _TraverseInternalsType
from .visitors import InternalTraversal
from .visitors import prefix_anon_map
from .. import exc
from .. import util
from ..util import HasMemoized_ro_memoized_attribute
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
class AliasedReturnsRows(NoInit, NamedFromClause):
    """Base class of aliases against tables, subqueries, and other
    selectables."""
    _is_from_container = True
    _supports_derived_columns = False
    element: ReturnsRows
    _traverse_internals: _TraverseInternalsType = [('element', InternalTraversal.dp_clauseelement), ('name', InternalTraversal.dp_anon_name)]

    @classmethod
    def _construct(cls, selectable: Any, *, name: Optional[str]=None, **kw: Any) -> Self:
        obj = cls.__new__(cls)
        obj._init(selectable, name=name, **kw)
        return obj

    def _init(self, selectable: Any, *, name: Optional[str]=None) -> None:
        self.element = coercions.expect(roles.ReturnsRowsRole, selectable, apply_propagate_attrs=self)
        self.element = selectable
        self._orig_name = name
        if name is None:
            if isinstance(selectable, FromClause) and selectable.named_with_column:
                name = getattr(selectable, 'name', None)
                if isinstance(name, _anonymous_label):
                    name = None
            name = _anonymous_label.safe_construct(id(self), name or 'anon')
        self.name = name

    def _refresh_for_new_column(self, column: ColumnElement[Any]) -> None:
        super()._refresh_for_new_column(column)
        self.element._refresh_for_new_column(column)

    def _populate_column_collection(self) -> None:
        self.element._generate_fromclause_column_proxies(self)

    @util.ro_non_memoized_property
    def description(self) -> str:
        name = self.name
        if isinstance(name, _anonymous_label):
            name = 'anon_1'
        return name

    @util.ro_non_memoized_property
    def implicit_returning(self) -> bool:
        return self.element.implicit_returning

    @property
    def original(self) -> ReturnsRows:
        """Legacy for dialects that are referring to Alias.original."""
        return self.element

    def is_derived_from(self, fromclause: Optional[FromClause]) -> bool:
        if fromclause in self._cloned_set:
            return True
        return self.element.is_derived_from(fromclause)

    def _copy_internals(self, clone: _CloneCallableType=_clone, **kw: Any) -> None:
        existing_element = self.element
        super()._copy_internals(clone=clone, **kw)
        if existing_element is not self.element:
            self._reset_column_collection()

    @property
    def _from_objects(self) -> List[FromClause]:
        return [self]