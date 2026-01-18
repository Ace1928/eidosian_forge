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
class CTE(roles.DMLTableRole, roles.IsCTERole, Generative, HasPrefixes, HasSuffixes, AliasedReturnsRows):
    """Represent a Common Table Expression.

    The :class:`_expression.CTE` object is obtained using the
    :meth:`_sql.SelectBase.cte` method from any SELECT statement. A less often
    available syntax also allows use of the :meth:`_sql.HasCTE.cte` method
    present on :term:`DML` constructs such as :class:`_sql.Insert`,
    :class:`_sql.Update` and
    :class:`_sql.Delete`.   See the :meth:`_sql.HasCTE.cte` method for
    usage details on CTEs.

    .. seealso::

        :ref:`tutorial_subqueries_ctes` - in the 2.0 tutorial

        :meth:`_sql.HasCTE.cte` - examples of calling styles

    """
    __visit_name__ = 'cte'
    _traverse_internals: _TraverseInternalsType = AliasedReturnsRows._traverse_internals + [('_cte_alias', InternalTraversal.dp_clauseelement), ('_restates', InternalTraversal.dp_clauseelement), ('recursive', InternalTraversal.dp_boolean), ('nesting', InternalTraversal.dp_boolean)] + HasPrefixes._has_prefixes_traverse_internals + HasSuffixes._has_suffixes_traverse_internals
    element: HasCTE

    @classmethod
    def _factory(cls, selectable: HasCTE, name: Optional[str]=None, recursive: bool=False) -> CTE:
        """Return a new :class:`_expression.CTE`,
        or Common Table Expression instance.

        Please see :meth:`_expression.HasCTE.cte` for detail on CTE usage.

        """
        return coercions.expect(roles.HasCTERole, selectable).cte(name=name, recursive=recursive)

    def _init(self, selectable: Select[Any], *, name: Optional[str]=None, recursive: bool=False, nesting: bool=False, _cte_alias: Optional[CTE]=None, _restates: Optional[CTE]=None, _prefixes: Optional[Tuple[()]]=None, _suffixes: Optional[Tuple[()]]=None) -> None:
        self.recursive = recursive
        self.nesting = nesting
        self._cte_alias = _cte_alias
        self._restates = _restates
        if _prefixes:
            self._prefixes = _prefixes
        if _suffixes:
            self._suffixes = _suffixes
        super()._init(selectable, name=name)

    def _populate_column_collection(self) -> None:
        if self._cte_alias is not None:
            self._cte_alias._generate_fromclause_column_proxies(self)
        else:
            self.element._generate_fromclause_column_proxies(self)

    def alias(self, name: Optional[str]=None, flat: bool=False) -> CTE:
        """Return an :class:`_expression.Alias` of this
        :class:`_expression.CTE`.

        This method is a CTE-specific specialization of the
        :meth:`_expression.FromClause.alias` method.

        .. seealso::

            :ref:`tutorial_using_aliases`

            :func:`_expression.alias`

        """
        return CTE._construct(self.element, name=name, recursive=self.recursive, nesting=self.nesting, _cte_alias=self, _prefixes=self._prefixes, _suffixes=self._suffixes)

    def union(self, *other: _SelectStatementForCompoundArgument) -> CTE:
        """Return a new :class:`_expression.CTE` with a SQL ``UNION``
        of the original CTE against the given selectables provided
        as positional arguments.

        :param \\*other: one or more elements with which to create a
         UNION.

         .. versionchanged:: 1.4.28 multiple elements are now accepted.

        .. seealso::

            :meth:`_sql.HasCTE.cte` - examples of calling styles

        """
        assert is_select_statement(self.element), f'CTE element f{self.element} does not support union()'
        return CTE._construct(self.element.union(*other), name=self.name, recursive=self.recursive, nesting=self.nesting, _restates=self, _prefixes=self._prefixes, _suffixes=self._suffixes)

    def union_all(self, *other: _SelectStatementForCompoundArgument) -> CTE:
        """Return a new :class:`_expression.CTE` with a SQL ``UNION ALL``
        of the original CTE against the given selectables provided
        as positional arguments.

        :param \\*other: one or more elements with which to create a
         UNION.

         .. versionchanged:: 1.4.28 multiple elements are now accepted.

        .. seealso::

            :meth:`_sql.HasCTE.cte` - examples of calling styles

        """
        assert is_select_statement(self.element), f'CTE element f{self.element} does not support union_all()'
        return CTE._construct(self.element.union_all(*other), name=self.name, recursive=self.recursive, nesting=self.nesting, _restates=self, _prefixes=self._prefixes, _suffixes=self._suffixes)

    def _get_reference_cte(self) -> CTE:
        """
        A recursive CTE is updated to attach the recursive part.
        Updated CTEs should still refer to the original CTE.
        This function returns this reference identifier.
        """
        return self._restates if self._restates is not None else self