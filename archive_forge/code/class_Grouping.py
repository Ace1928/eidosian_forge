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
class Grouping(GroupedElement, ColumnElement[_T]):
    """Represent a grouping within a column expression"""
    _traverse_internals: _TraverseInternalsType = [('element', InternalTraversal.dp_clauseelement), ('type', InternalTraversal.dp_type)]
    _cache_key_traversal = [('element', InternalTraversal.dp_clauseelement)]
    element: Union[TextClause, ClauseList, ColumnElement[_T]]

    def __init__(self, element: Union[TextClause, ClauseList, ColumnElement[_T]]):
        self.element = element
        self.type = getattr(element, 'type', type_api.NULLTYPE)
        self._propagate_attrs = element._propagate_attrs

    def _with_binary_element_type(self, type_):
        return self.__class__(self.element._with_binary_element_type(type_))

    @util.memoized_property
    def _is_implicitly_boolean(self):
        return self.element._is_implicitly_boolean

    @util.non_memoized_property
    def _tq_label(self) -> Optional[str]:
        return getattr(self.element, '_tq_label', None) or self._anon_name_label

    @util.non_memoized_property
    def _proxies(self) -> List[ColumnElement[Any]]:
        if isinstance(self.element, ColumnElement):
            return [self.element]
        else:
            return []

    @util.ro_non_memoized_property
    def _from_objects(self) -> List[FromClause]:
        return self.element._from_objects

    def __getattr__(self, attr):
        return getattr(self.element, attr)

    def __getstate__(self):
        return {'element': self.element, 'type': self.type}

    def __setstate__(self, state):
        self.element = state['element']
        self.type = state['type']