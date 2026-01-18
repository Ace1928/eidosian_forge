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
class ScalarSelect(roles.InElementRole, Generative, GroupedElement, ColumnElement[_T]):
    """Represent a scalar subquery.


    A :class:`_sql.ScalarSelect` is created by invoking the
    :meth:`_sql.SelectBase.scalar_subquery` method.   The object
    then participates in other SQL expressions as a SQL column expression
    within the :class:`_sql.ColumnElement` hierarchy.

    .. seealso::

        :meth:`_sql.SelectBase.scalar_subquery`

        :ref:`tutorial_scalar_subquery` - in the 2.0 tutorial

    """
    _traverse_internals: _TraverseInternalsType = [('element', InternalTraversal.dp_clauseelement), ('type', InternalTraversal.dp_type)]
    _from_objects: List[FromClause] = []
    _is_from_container = True
    if not TYPE_CHECKING:
        _is_implicitly_boolean = False
    inherit_cache = True
    element: SelectBase

    def __init__(self, element: SelectBase) -> None:
        self.element = element
        self.type = element._scalar_type()
        self._propagate_attrs = element._propagate_attrs

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.element, attr)

    def __getstate__(self) -> Dict[str, Any]:
        return {'element': self.element, 'type': self.type}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.element = state['element']
        self.type = state['type']

    @property
    def columns(self) -> NoReturn:
        raise exc.InvalidRequestError('Scalar Select expression has no columns; use this object directly within a column-level expression.')
    c = columns

    @_generative
    def where(self, crit: _ColumnExpressionArgument[bool]) -> Self:
        """Apply a WHERE clause to the SELECT statement referred to
        by this :class:`_expression.ScalarSelect`.

        """
        self.element = cast('Select[Any]', self.element).where(crit)
        return self

    @overload
    def self_group(self: ScalarSelect[Any], against: Optional[OperatorType]=None) -> ScalarSelect[Any]:
        ...

    @overload
    def self_group(self: ColumnElement[Any], against: Optional[OperatorType]=None) -> ColumnElement[Any]:
        ...

    def self_group(self, against: Optional[OperatorType]=None) -> ColumnElement[Any]:
        return self
    if TYPE_CHECKING:

        def _ungroup(self) -> Select[Any]:
            ...

    @_generative
    def correlate(self, *fromclauses: Union[Literal[None, False], _FromClauseArgument]) -> Self:
        """Return a new :class:`_expression.ScalarSelect`
        which will correlate the given FROM
        clauses to that of an enclosing :class:`_expression.Select`.

        This method is mirrored from the :meth:`_sql.Select.correlate` method
        of the underlying :class:`_sql.Select`.  The method applies the
        :meth:_sql.Select.correlate` method, then returns a new
        :class:`_sql.ScalarSelect` against that statement.

        .. versionadded:: 1.4 Previously, the
           :meth:`_sql.ScalarSelect.correlate`
           method was only available from :class:`_sql.Select`.

        :param \\*fromclauses: a list of one or more
         :class:`_expression.FromClause`
         constructs, or other compatible constructs (i.e. ORM-mapped
         classes) to become part of the correlate collection.

        .. seealso::

            :meth:`_expression.ScalarSelect.correlate_except`

            :ref:`tutorial_scalar_subquery` - in the 2.0 tutorial


        """
        self.element = cast('Select[Any]', self.element).correlate(*fromclauses)
        return self

    @_generative
    def correlate_except(self, *fromclauses: Union[Literal[None, False], _FromClauseArgument]) -> Self:
        """Return a new :class:`_expression.ScalarSelect`
        which will omit the given FROM
        clauses from the auto-correlation process.

        This method is mirrored from the
        :meth:`_sql.Select.correlate_except` method of the underlying
        :class:`_sql.Select`.  The method applies the
        :meth:_sql.Select.correlate_except` method, then returns a new
        :class:`_sql.ScalarSelect` against that statement.

        .. versionadded:: 1.4 Previously, the
           :meth:`_sql.ScalarSelect.correlate_except`
           method was only available from :class:`_sql.Select`.

        :param \\*fromclauses: a list of one or more
         :class:`_expression.FromClause`
         constructs, or other compatible constructs (i.e. ORM-mapped
         classes) to become part of the correlate-exception collection.

        .. seealso::

            :meth:`_expression.ScalarSelect.correlate`

            :ref:`tutorial_scalar_subquery` - in the 2.0 tutorial


        """
        self.element = cast('Select[Any]', self.element).correlate_except(*fromclauses)
        return self