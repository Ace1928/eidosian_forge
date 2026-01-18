from __future__ import annotations
import enum
import functools
import re
import types
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
from typing import Match
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes  # noqa
from . import exc
from ._typing import _O
from ._typing import insp_is_aliased_class
from ._typing import insp_is_mapper
from ._typing import prop_is_relationship
from .base import _class_to_mapper as _class_to_mapper
from .base import _MappedAnnotationBase
from .base import _never_set as _never_set  # noqa: F401
from .base import _none_set as _none_set  # noqa: F401
from .base import attribute_str as attribute_str  # noqa: F401
from .base import class_mapper as class_mapper
from .base import DynamicMapped
from .base import InspectionAttr as InspectionAttr
from .base import instance_str as instance_str  # noqa: F401
from .base import Mapped
from .base import object_mapper as object_mapper
from .base import object_state as object_state  # noqa: F401
from .base import opt_manager_of_class
from .base import ORMDescriptor
from .base import state_attribute_str as state_attribute_str  # noqa: F401
from .base import state_class_str as state_class_str  # noqa: F401
from .base import state_str as state_str  # noqa: F401
from .base import WriteOnlyMapped
from .interfaces import CriteriaOption
from .interfaces import MapperProperty as MapperProperty
from .interfaces import ORMColumnsClauseRole
from .interfaces import ORMEntityColumnsClauseRole
from .interfaces import ORMFromClauseRole
from .path_registry import PathRegistry as PathRegistry
from .. import event
from .. import exc as sa_exc
from .. import inspection
from .. import sql
from .. import util
from ..engine.result import result_tuple
from ..sql import coercions
from ..sql import expression
from ..sql import lambdas
from ..sql import roles
from ..sql import util as sql_util
from ..sql import visitors
from ..sql._typing import is_selectable
from ..sql.annotation import SupportsCloneAnnotations
from ..sql.base import ColumnCollection
from ..sql.cache_key import HasCacheKey
from ..sql.cache_key import MemoizedHasCacheKey
from ..sql.elements import ColumnElement
from ..sql.elements import KeyedColumnElement
from ..sql.selectable import FromClause
from ..util.langhelpers import MemoizedSlots
from ..util.typing import de_stringify_annotation as _de_stringify_annotation
from ..util.typing import (
from ..util.typing import eval_name_only as _eval_name_only
from ..util.typing import is_origin_of_cls
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import typing_get_origin
class _TraceAdaptRole(enum.Enum):
    """Enumeration of all the use cases for ORMAdapter.

    ORMAdapter remains one of the most complicated aspects of the ORM, as it is
    used for in-place adaption of column expressions to be applied to a SELECT,
    replacing :class:`.Table` and other objects that are mapped to classes with
    aliases of those tables in the case of joined eager loading, or in the case
    of polymorphic loading as used with concrete mappings or other custom "with
    polymorphic" parameters, with whole user-defined subqueries. The
    enumerations provide an overview of all the use cases used by ORMAdapter, a
    layer of formality as to the introduction of new ORMAdapter use cases (of
    which none are anticipated), as well as a means to trace the origins of a
    particular ORMAdapter within runtime debugging.

    SQLAlchemy 2.0 has greatly scaled back ORM features which relied heavily on
    open-ended statement adaption, including the ``Query.with_polymorphic()``
    method and the ``Query.select_from_entity()`` methods, favoring
    user-explicit aliasing schemes using the ``aliased()`` and
    ``with_polymorphic()`` standalone constructs; these still use adaption,
    however the adaption is applied in a narrower scope.

    """
    ALIASED_INSP = enum.auto()
    JOINEDLOAD_USER_DEFINED_ALIAS = enum.auto()
    JOINEDLOAD_PATH_WITH_POLYMORPHIC = enum.auto()
    JOINEDLOAD_MEMOIZED_ADAPTER = enum.auto()
    MAPPER_POLYMORPHIC_ADAPTER = enum.auto()
    WITH_POLYMORPHIC_ADAPTER = enum.auto()
    WITH_POLYMORPHIC_ADAPTER_RIGHT_JOIN = enum.auto()
    DEPRECATED_JOIN_ADAPT_RIGHT_SIDE = enum.auto()
    ADAPT_FROM_STATEMENT = enum.auto()
    COMPOUND_EAGER_STATEMENT = enum.auto()
    LEGACY_SELECT_FROM_ALIAS = enum.auto()