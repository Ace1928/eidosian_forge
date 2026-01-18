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
class AnnotatedColumnElement(Annotated):
    _Annotated__element: ColumnElement[Any]

    def __init__(self, element, values):
        Annotated.__init__(self, element, values)
        for attr in ('comparator', '_proxy_key', '_tq_key_label', '_tq_label', '_non_anon_label', 'type'):
            self.__dict__.pop(attr, None)
        for attr in ('name', 'key', 'table'):
            if self.__dict__.get(attr, False) is None:
                self.__dict__.pop(attr)

    def _with_annotations(self, values):
        clone = super()._with_annotations(values)
        clone.__dict__.pop('comparator', None)
        return clone

    @util.memoized_property
    def name(self):
        """pull 'name' from parent, if not present"""
        return self._Annotated__element.name

    @_memoized_property_but_not_nulltype
    def type(self):
        """pull 'type' from parent and don't cache if null.

        type is routinely changed on existing columns within the
        mapped_column() initialization process, and "type" is also consulted
        during the creation of SQL expressions.  Therefore it can change after
        it was already retrieved.  At the same time we don't want annotated
        objects having overhead when expressions are produced, so continue
        to memoize, but only when we have a non-null type.

        """
        return self._Annotated__element.type

    @util.memoized_property
    def table(self):
        """pull 'table' from parent, if not present"""
        return self._Annotated__element.table

    @util.memoized_property
    def key(self):
        """pull 'key' from parent, if not present"""
        return self._Annotated__element.key

    @util.memoized_property
    def info(self) -> _InfoType:
        if TYPE_CHECKING:
            assert isinstance(self._Annotated__element, Column)
        return self._Annotated__element.info

    @util.memoized_property
    def _anon_name_label(self) -> str:
        return self._Annotated__element._anon_name_label