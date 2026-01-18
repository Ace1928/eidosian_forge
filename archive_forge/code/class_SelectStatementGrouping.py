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
class SelectStatementGrouping(GroupedElement, SelectBase, Generic[_SB]):
    """Represent a grouping of a :class:`_expression.SelectBase`.

    This differs from :class:`.Subquery` in that we are still
    an "inner" SELECT statement, this is strictly for grouping inside of
    compound selects.

    """
    __visit_name__ = 'select_statement_grouping'
    _traverse_internals: _TraverseInternalsType = [('element', InternalTraversal.dp_clauseelement)]
    _is_select_container = True
    element: _SB

    def __init__(self, element: _SB) -> None:
        self.element = cast(_SB, coercions.expect(roles.SelectStatementRole, element))

    def _ensure_disambiguated_names(self) -> SelectStatementGrouping[_SB]:
        new_element = self.element._ensure_disambiguated_names()
        if new_element is not self.element:
            return SelectStatementGrouping(new_element)
        else:
            return self

    def get_label_style(self) -> SelectLabelStyle:
        return self.element.get_label_style()

    def set_label_style(self, label_style: SelectLabelStyle) -> SelectStatementGrouping[_SB]:
        return SelectStatementGrouping(self.element.set_label_style(label_style))

    @property
    def select_statement(self) -> _SB:
        return self.element

    def self_group(self, against: Optional[OperatorType]=None) -> Self:
        ...
        return self
    if TYPE_CHECKING:

        def _ungroup(self) -> _SB:
            ...

    def _generate_fromclause_column_proxies(self, subquery: FromClause, *, proxy_compound_columns: Optional[Iterable[Sequence[ColumnElement[Any]]]]=None) -> None:
        self.element._generate_fromclause_column_proxies(subquery, proxy_compound_columns=proxy_compound_columns)

    @util.ro_non_memoized_property
    def _all_selected_columns(self) -> _SelectIterable:
        return self.element._all_selected_columns

    @util.ro_non_memoized_property
    def selected_columns(self) -> ColumnCollection[str, ColumnElement[Any]]:
        """A :class:`_expression.ColumnCollection`
        representing the columns that
        the embedded SELECT statement returns in its result set, not including
        :class:`_sql.TextClause` constructs.

        .. versionadded:: 1.4

        .. seealso::

            :attr:`_sql.Select.selected_columns`

        """
        return self.element.selected_columns

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return self.element._from_objects