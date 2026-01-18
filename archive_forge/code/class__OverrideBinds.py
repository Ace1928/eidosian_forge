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
class _OverrideBinds(Grouping[_T]):
    """used by cache_key->_apply_params_to_element to allow compilation /
    execution of a SQL element that's been cached, using an alternate set of
    bound parameter values.

    This is used by the ORM to swap new parameter values into expressions
    that are embedded into loader options like with_expression(),
    selectinload().  Previously, this task was accomplished using the
    .params() method which would perform a deep-copy instead.  This deep
    copy proved to be too expensive for more complex expressions.

    See #11085

    """
    __visit_name__ = 'override_binds'

    def __init__(self, element: ColumnElement[_T], bindparams: Sequence[BindParameter[Any]], replaces_params: Sequence[BindParameter[Any]]):
        self.element = element
        self.translate = {k.key: v.value for k, v in zip(replaces_params, bindparams)}

    def _gen_cache_key(self, anon_map: anon_map, bindparams: List[BindParameter[Any]]) -> Optional[typing_Tuple[Any, ...]]:
        """generate a cache key for the given element, substituting its bind
        values for the translation values present."""
        existing_bps: List[BindParameter[Any]] = []
        ck = self.element._gen_cache_key(anon_map, existing_bps)
        bindparams.extend((bp._with_value(self.translate[bp.key], maintain_key=True, required=False) if bp.key in self.translate else bp for bp in existing_bps))
        return ck