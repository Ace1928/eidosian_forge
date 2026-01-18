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
class WrapsColumnExpression(ColumnElement[_T]):
    """Mixin that defines a :class:`_expression.ColumnElement`
    as a wrapper with special
    labeling behavior for an expression that already has a name.

    .. versionadded:: 1.4

    .. seealso::

        :ref:`change_4449`


    """

    @property
    def wrapped_column_expression(self) -> ColumnElement[_T]:
        raise NotImplementedError()

    @util.non_memoized_property
    def _tq_label(self) -> Optional[str]:
        wce = self.wrapped_column_expression
        if hasattr(wce, '_tq_label'):
            return wce._tq_label
        else:
            return None

    @property
    def _label(self) -> Optional[str]:
        return self._tq_label

    @property
    def _non_anon_label(self) -> Optional[str]:
        return None

    @util.non_memoized_property
    def _anon_name_label(self) -> str:
        wce = self.wrapped_column_expression
        if not wce._is_text_clause:
            nal = wce._non_anon_label
            if nal:
                return nal
            elif hasattr(wce, '_anon_name_label'):
                return wce._anon_name_label
        return super()._anon_name_label

    def _dedupe_anon_label_idx(self, idx: int) -> str:
        wce = self.wrapped_column_expression
        nal = wce._non_anon_label
        if nal:
            return self._anon_label(nal + '_')
        else:
            return self._dedupe_anon_tq_label_idx(idx)

    @property
    def _proxy_key(self):
        wce = self.wrapped_column_expression
        if not wce._is_text_clause:
            return wce._proxy_key
        return super()._proxy_key