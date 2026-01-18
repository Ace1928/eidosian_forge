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
class Over(ColumnElement[_T]):
    """Represent an OVER clause.

    This is a special operator against a so-called
    "window" function, as well as any aggregate function,
    which produces results relative to the result set
    itself.  Most modern SQL backends now support window functions.

    """
    __visit_name__ = 'over'
    _traverse_internals: _TraverseInternalsType = [('element', InternalTraversal.dp_clauseelement), ('order_by', InternalTraversal.dp_clauseelement), ('partition_by', InternalTraversal.dp_clauseelement), ('range_', InternalTraversal.dp_plain_obj), ('rows', InternalTraversal.dp_plain_obj)]
    order_by: Optional[ClauseList] = None
    partition_by: Optional[ClauseList] = None
    element: ColumnElement[_T]
    'The underlying expression object to which this :class:`.Over`\n    object refers.'
    range_: Optional[typing_Tuple[int, int]]

    def __init__(self, element: ColumnElement[_T], partition_by: Optional[_ByArgument]=None, order_by: Optional[_ByArgument]=None, range_: Optional[typing_Tuple[Optional[int], Optional[int]]]=None, rows: Optional[typing_Tuple[Optional[int], Optional[int]]]=None):
        self.element = element
        if order_by is not None:
            self.order_by = ClauseList(*util.to_list(order_by), _literal_as_text_role=roles.ByOfRole)
        if partition_by is not None:
            self.partition_by = ClauseList(*util.to_list(partition_by), _literal_as_text_role=roles.ByOfRole)
        if range_:
            self.range_ = self._interpret_range(range_)
            if rows:
                raise exc.ArgumentError("'range_' and 'rows' are mutually exclusive")
            else:
                self.rows = None
        elif rows:
            self.rows = self._interpret_range(rows)
            self.range_ = None
        else:
            self.rows = self.range_ = None

    def __reduce__(self):
        return (self.__class__, (self.element, self.partition_by, self.order_by, self.range_, self.rows))

    def _interpret_range(self, range_: typing_Tuple[Optional[int], Optional[int]]) -> typing_Tuple[int, int]:
        if not isinstance(range_, tuple) or len(range_) != 2:
            raise exc.ArgumentError('2-tuple expected for range/rows')
        lower: int
        upper: int
        if range_[0] is None:
            lower = RANGE_UNBOUNDED
        else:
            try:
                lower = int(range_[0])
            except ValueError as err:
                raise exc.ArgumentError('Integer or None expected for range value') from err
            else:
                if lower == 0:
                    lower = RANGE_CURRENT
        if range_[1] is None:
            upper = RANGE_UNBOUNDED
        else:
            try:
                upper = int(range_[1])
            except ValueError as err:
                raise exc.ArgumentError('Integer or None expected for range value') from err
            else:
                if upper == 0:
                    upper = RANGE_CURRENT
        return (lower, upper)
    if not TYPE_CHECKING:

        @util.memoized_property
        def type(self) -> TypeEngine[_T]:
            return self.element.type

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return list(itertools.chain(*[c._from_objects for c in (self.element, self.partition_by, self.order_by) if c is not None]))