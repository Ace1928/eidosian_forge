from __future__ import annotations
import collections
import dataclasses
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Generic
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import exc as orm_exc
from . import path_registry
from .base import _MappedAttribute as _MappedAttribute
from .base import EXT_CONTINUE as EXT_CONTINUE  # noqa: F401
from .base import EXT_SKIP as EXT_SKIP  # noqa: F401
from .base import EXT_STOP as EXT_STOP  # noqa: F401
from .base import InspectionAttr as InspectionAttr  # noqa: F401
from .base import InspectionAttrInfo as InspectionAttrInfo
from .base import MANYTOMANY as MANYTOMANY  # noqa: F401
from .base import MANYTOONE as MANYTOONE  # noqa: F401
from .base import NO_KEY as NO_KEY  # noqa: F401
from .base import NO_VALUE as NO_VALUE  # noqa: F401
from .base import NotExtension as NotExtension  # noqa: F401
from .base import ONETOMANY as ONETOMANY  # noqa: F401
from .base import RelationshipDirection as RelationshipDirection  # noqa: F401
from .base import SQLORMOperations
from .. import ColumnElement
from .. import exc as sa_exc
from .. import inspection
from .. import util
from ..sql import operators
from ..sql import roles
from ..sql import visitors
from ..sql.base import _NoArg
from ..sql.base import ExecutableOption
from ..sql.cache_key import HasCacheKey
from ..sql.operators import ColumnOperators
from ..sql.schema import Column
from ..sql.type_api import TypeEngine
from ..util import warn_deprecated
from ..util.typing import RODescriptorReference
from ..util.typing import TypedDict
class CompileStateOption(HasCacheKey, ORMOption):
    """base for :class:`.ORMOption` classes that affect the compilation of
    a SQL query and therefore need to be part of the cache key.

    .. note::  :class:`.CompileStateOption` is generally non-public and
       should not be used as a base class for user-defined options; instead,
       use :class:`.UserDefinedOption`, which is easier to use as it does not
       interact with ORM compilation internals or caching.

    :class:`.CompileStateOption` defines an internal attribute
    ``_is_compile_state=True`` which has the effect of the ORM compilation
    routines for SELECT and other statements will call upon these options when
    a SQL string is being compiled. As such, these classes implement
    :class:`.HasCacheKey` and need to provide robust ``_cache_key_traversal``
    structures.

    The :class:`.CompileStateOption` class is used to implement the ORM
    :class:`.LoaderOption` and :class:`.CriteriaOption` classes.

    .. versionadded:: 1.4.28


    """
    __slots__ = ()
    _is_compile_state = True

    def process_compile_state(self, compile_state: ORMCompileState) -> None:
        """Apply a modification to a given :class:`.ORMCompileState`.

        This method is part of the implementation of a particular
        :class:`.CompileStateOption` and is only invoked internally
        when an ORM query is compiled.

        """

    def process_compile_state_replaced_entities(self, compile_state: ORMCompileState, mapper_entities: Sequence[_MapperEntity]) -> None:
        """Apply a modification to a given :class:`.ORMCompileState`,
        given entities that were replaced by with_only_columns() or
        with_entities().

        This method is part of the implementation of a particular
        :class:`.CompileStateOption` and is only invoked internally
        when an ORM query is compiled.

        .. versionadded:: 1.4.19

        """