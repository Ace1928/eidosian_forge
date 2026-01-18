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
class ReturnsRows(roles.ReturnsRowsRole, DQLDMLClauseElement):
    """The base-most class for Core constructs that have some concept of
    columns that can represent rows.

    While the SELECT statement and TABLE are the primary things we think
    of in this category,  DML like INSERT, UPDATE and DELETE can also specify
    RETURNING which means they can be used in CTEs and other forms, and
    PostgreSQL has functions that return rows also.

    .. versionadded:: 1.4

    """
    _is_returns_rows = True
    _is_from_clause = False
    _is_select_base = False
    _is_select_statement = False
    _is_lateral = False

    @property
    def selectable(self) -> ReturnsRows:
        return self

    @util.ro_non_memoized_property
    def _all_selected_columns(self) -> _SelectIterable:
        """A sequence of column expression objects that represents the
        "selected" columns of this :class:`_expression.ReturnsRows`.

        This is typically equivalent to .exported_columns except it is
        delivered in the form of a straight sequence and not  keyed
        :class:`_expression.ColumnCollection`.

        """
        raise NotImplementedError()

    def is_derived_from(self, fromclause: Optional[FromClause]) -> bool:
        """Return ``True`` if this :class:`.ReturnsRows` is
        'derived' from the given :class:`.FromClause`.

        An example would be an Alias of a Table is derived from that Table.

        """
        raise NotImplementedError()

    def _generate_fromclause_column_proxies(self, fromclause: FromClause) -> None:
        """Populate columns into an :class:`.AliasedReturnsRows` object."""
        raise NotImplementedError()

    def _refresh_for_new_column(self, column: ColumnElement[Any]) -> None:
        """reset internal collections for an incoming column being added."""
        raise NotImplementedError()

    @property
    def exported_columns(self) -> ReadOnlyColumnCollection[Any, Any]:
        """A :class:`_expression.ColumnCollection`
        that represents the "exported"
        columns of this :class:`_expression.ReturnsRows`.

        The "exported" columns represent the collection of
        :class:`_expression.ColumnElement`
        expressions that are rendered by this SQL
        construct.   There are primary varieties which are the
        "FROM clause columns" of a FROM clause, such as a table, join,
        or subquery, the "SELECTed columns", which are the columns in
        the "columns clause" of a SELECT statement, and the RETURNING
        columns in a DML statement..

        .. versionadded:: 1.4

        .. seealso::

            :attr:`_expression.FromClause.exported_columns`

            :attr:`_expression.SelectBase.exported_columns`
        """
        raise NotImplementedError()