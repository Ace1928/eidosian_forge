from __future__ import annotations
from enum import Enum
from types import ModuleType
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Mapping
from typing import NewType
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .base import SchemaEventTarget
from .cache_key import CacheConst
from .cache_key import NO_CACHE
from .operators import ColumnOperators
from .visitors import Visitable
from .. import exc
from .. import util
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypeAliasType
from ..util.typing import TypedDict
from ..util.typing import TypeGuard
class TypeEngineMixin:
    """classes which subclass this can act as "mixin" classes for
    TypeEngine."""
    __slots__ = ()
    if TYPE_CHECKING:

        @util.memoized_property
        def _static_cache_key(self) -> Union[CacheConst, Tuple[Any, ...]]:
            ...

        @overload
        def adapt(self, cls: Type[_TE], **kw: Any) -> _TE:
            ...

        @overload
        def adapt(self, cls: Type[TypeEngineMixin], **kw: Any) -> TypeEngine[Any]:
            ...

        def adapt(self, cls: Type[Union[TypeEngine[Any], TypeEngineMixin]], **kw: Any) -> TypeEngine[Any]:
            ...

        def dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
            ...