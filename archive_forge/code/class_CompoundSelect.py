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
class CompoundSelect(HasCompileState, GenerativeSelect, ExecutableReturnsRows):
    """Forms the basis of ``UNION``, ``UNION ALL``, and other
    SELECT-based set operations.


    .. seealso::

        :func:`_expression.union`

        :func:`_expression.union_all`

        :func:`_expression.intersect`

        :func:`_expression.intersect_all`

        :func:`_expression.except`

        :func:`_expression.except_all`

    """
    __visit_name__ = 'compound_select'
    _traverse_internals: _TraverseInternalsType = [('selects', InternalTraversal.dp_clauseelement_list), ('_limit_clause', InternalTraversal.dp_clauseelement), ('_offset_clause', InternalTraversal.dp_clauseelement), ('_fetch_clause', InternalTraversal.dp_clauseelement), ('_fetch_clause_options', InternalTraversal.dp_plain_dict), ('_order_by_clauses', InternalTraversal.dp_clauseelement_list), ('_group_by_clauses', InternalTraversal.dp_clauseelement_list), ('_for_update_arg', InternalTraversal.dp_clauseelement), ('keyword', InternalTraversal.dp_string)] + SupportsCloneAnnotations._clone_annotations_traverse_internals
    selects: List[SelectBase]
    _is_from_container = True
    _auto_correlate = False

    def __init__(self, keyword: _CompoundSelectKeyword, *selects: _SelectStatementForCompoundArgument):
        self.keyword = keyword
        self.selects = [coercions.expect(roles.CompoundElementRole, s, apply_propagate_attrs=self).self_group(against=self) for s in selects]
        GenerativeSelect.__init__(self)

    @classmethod
    def _create_union(cls, *selects: _SelectStatementForCompoundArgument) -> CompoundSelect:
        return CompoundSelect(_CompoundSelectKeyword.UNION, *selects)

    @classmethod
    def _create_union_all(cls, *selects: _SelectStatementForCompoundArgument) -> CompoundSelect:
        return CompoundSelect(_CompoundSelectKeyword.UNION_ALL, *selects)

    @classmethod
    def _create_except(cls, *selects: _SelectStatementForCompoundArgument) -> CompoundSelect:
        return CompoundSelect(_CompoundSelectKeyword.EXCEPT, *selects)

    @classmethod
    def _create_except_all(cls, *selects: _SelectStatementForCompoundArgument) -> CompoundSelect:
        return CompoundSelect(_CompoundSelectKeyword.EXCEPT_ALL, *selects)

    @classmethod
    def _create_intersect(cls, *selects: _SelectStatementForCompoundArgument) -> CompoundSelect:
        return CompoundSelect(_CompoundSelectKeyword.INTERSECT, *selects)

    @classmethod
    def _create_intersect_all(cls, *selects: _SelectStatementForCompoundArgument) -> CompoundSelect:
        return CompoundSelect(_CompoundSelectKeyword.INTERSECT_ALL, *selects)

    def _scalar_type(self) -> TypeEngine[Any]:
        return self.selects[0]._scalar_type()

    def self_group(self, against: Optional[OperatorType]=None) -> GroupedElement:
        return SelectStatementGrouping(self)

    def is_derived_from(self, fromclause: Optional[FromClause]) -> bool:
        for s in self.selects:
            if s.is_derived_from(fromclause):
                return True
        return False

    def set_label_style(self, style: SelectLabelStyle) -> CompoundSelect:
        if self._label_style is not style:
            self = self._generate()
            select_0 = self.selects[0].set_label_style(style)
            self.selects = [select_0] + self.selects[1:]
        return self

    def _ensure_disambiguated_names(self) -> CompoundSelect:
        new_select = self.selects[0]._ensure_disambiguated_names()
        if new_select is not self.selects[0]:
            self = self._generate()
            self.selects = [new_select] + self.selects[1:]
        return self

    def _generate_fromclause_column_proxies(self, subquery: FromClause, *, proxy_compound_columns: Optional[Iterable[Sequence[ColumnElement[Any]]]]=None) -> None:
        select_0 = self.selects[0]
        if self._label_style is not LABEL_STYLE_DEFAULT:
            select_0 = select_0.set_label_style(self._label_style)
        extra_col_iterator = zip(*[[c._annotate(dd) for c in stmt._all_selected_columns if is_column_element(c)] for dd, stmt in [({'weight': i + 1}, stmt) for i, stmt in enumerate(self.selects)]])
        select_0._generate_fromclause_column_proxies(subquery, proxy_compound_columns=extra_col_iterator)

    def _refresh_for_new_column(self, column: ColumnElement[Any]) -> None:
        super()._refresh_for_new_column(column)
        for select in self.selects:
            select._refresh_for_new_column(column)

    @util.ro_non_memoized_property
    def _all_selected_columns(self) -> _SelectIterable:
        return self.selects[0]._all_selected_columns

    @util.ro_non_memoized_property
    def selected_columns(self) -> ColumnCollection[str, ColumnElement[Any]]:
        """A :class:`_expression.ColumnCollection`
        representing the columns that
        this SELECT statement or similar construct returns in its result set,
        not including :class:`_sql.TextClause` constructs.

        For a :class:`_expression.CompoundSelect`, the
        :attr:`_expression.CompoundSelect.selected_columns`
        attribute returns the selected
        columns of the first SELECT statement contained within the series of
        statements within the set operation.

        .. seealso::

            :attr:`_sql.Select.selected_columns`

        .. versionadded:: 1.4

        """
        return self.selects[0].selected_columns