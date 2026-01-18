from __future__ import annotations
from decimal import Decimal
from enum import IntEnum
import itertools
import operator
import re
import typing
from typing import AbstractSet
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
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple as typing_Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import operators
from . import roles
from . import traversals
from . import type_api
from ._typing import has_schema_attr
from ._typing import is_named_from_clause
from ._typing import is_quoted_name
from ._typing import is_tuple_type
from .annotation import Annotated
from .annotation import SupportsWrappingAnnotations
from .base import _clone
from .base import _expand_cloned
from .base import _generative
from .base import _NoArg
from .base import Executable
from .base import Generative
from .base import HasMemoized
from .base import Immutable
from .base import NO_ARG
from .base import SingletonConstant
from .cache_key import MemoizedHasCacheKey
from .cache_key import NO_CACHE
from .coercions import _document_text_coercion  # noqa
from .operators import ColumnOperators
from .traversals import HasCopyInternals
from .visitors import cloned_traverse
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .visitors import traverse
from .visitors import Visitable
from .. import exc
from .. import inspection
from .. import util
from ..util import HasMemoized_ro_memoized_attribute
from ..util import TypingOnly
from ..util.typing import Literal
from ..util.typing import Self
class WithinGroup(ColumnElement[_T]):
    """Represent a WITHIN GROUP (ORDER BY) clause.

    This is a special operator against so-called
    "ordered set aggregate" and "hypothetical
    set aggregate" functions, including ``percentile_cont()``,
    ``rank()``, ``dense_rank()``, etc.

    It's supported only by certain database backends, such as PostgreSQL,
    Oracle and MS SQL Server.

    The :class:`.WithinGroup` construct extracts its type from the
    method :meth:`.FunctionElement.within_group_type`.  If this returns
    ``None``, the function's ``.type`` is used.

    """
    __visit_name__ = 'withingroup'
    _traverse_internals: _TraverseInternalsType = [('element', InternalTraversal.dp_clauseelement), ('order_by', InternalTraversal.dp_clauseelement)]
    order_by: Optional[ClauseList] = None

    def __init__(self, element: FunctionElement[_T], *order_by: _ColumnExpressionArgument[Any]):
        self.element = element
        if order_by is not None:
            self.order_by = ClauseList(*util.to_list(order_by), _literal_as_text_role=roles.ByOfRole)

    def __reduce__(self):
        return (self.__class__, (self.element,) + (tuple(self.order_by) if self.order_by is not None else ()))

    def over(self, partition_by=None, order_by=None, range_=None, rows=None):
        """Produce an OVER clause against this :class:`.WithinGroup`
        construct.

        This function has the same signature as that of
        :meth:`.FunctionElement.over`.

        """
        return Over(self, partition_by=partition_by, order_by=order_by, range_=range_, rows=rows)
    if not TYPE_CHECKING:

        @util.memoized_property
        def type(self) -> TypeEngine[_T]:
            wgt = self.element.within_group_type(self)
            if wgt is not None:
                return wgt
            else:
                return self.element.type

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return list(itertools.chain(*[c._from_objects for c in (self.element, self.order_by) if c is not None]))