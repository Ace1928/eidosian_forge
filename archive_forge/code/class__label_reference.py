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
class _label_reference(ColumnElement[_T]):
    """Wrap a column expression as it appears in a 'reference' context.

    This expression is any that includes an _order_by_label_element,
    which is a Label, or a DESC / ASC construct wrapping a Label.

    The production of _label_reference() should occur when an expression
    is added to this context; this includes the ORDER BY or GROUP BY of a
    SELECT statement, as well as a few other places, such as the ORDER BY
    within an OVER clause.

    """
    __visit_name__ = 'label_reference'
    _traverse_internals: _TraverseInternalsType = [('element', InternalTraversal.dp_clauseelement)]
    element: ColumnElement[_T]

    def __init__(self, element: ColumnElement[_T]):
        self.element = element

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return []