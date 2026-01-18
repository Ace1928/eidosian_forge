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
class Emulated(TypeEngineMixin):
    """Mixin for base types that emulate the behavior of a DB-native type.

    An :class:`.Emulated` type will use an available database type
    in conjunction with Python-side routines and/or database constraints
    in order to approximate the behavior of a database type that is provided
    natively by some backends.  When a native-providing backend is in
    use, the native version of the type is used.  This native version
    should include the :class:`.NativeForEmulated` mixin to allow it to be
    distinguished from :class:`.Emulated`.

    Current examples of :class:`.Emulated` are:  :class:`.Interval`,
    :class:`.Enum`, :class:`.Boolean`.

    .. versionadded:: 1.2.0b3

    """
    native: bool

    def adapt_to_emulated(self, impltype: Type[Union[TypeEngine[Any], TypeEngineMixin]], **kw: Any) -> TypeEngine[Any]:
        """Given an impl class, adapt this type to the impl assuming
        "emulated".

        The impl should also be an "emulated" version of this type,
        most likely the same class as this type itself.

        e.g.: sqltypes.Enum adapts to the Enum class.

        """
        return super().adapt(impltype, **kw)

    @overload
    def adapt(self, cls: Type[_TE], **kw: Any) -> _TE:
        ...

    @overload
    def adapt(self, cls: Type[TypeEngineMixin], **kw: Any) -> TypeEngine[Any]:
        ...

    def adapt(self, cls: Type[Union[TypeEngine[Any], TypeEngineMixin]], **kw: Any) -> TypeEngine[Any]:
        if _is_native_for_emulated(cls):
            if self.native:
                return cls.adapt_emulated_to_native(self, **kw)
            else:
                return cls.adapt_native_to_emulated(self, **kw)
        elif issubclass(cls, self.__class__):
            return self.adapt_to_emulated(cls, **kw)
        else:
            return super().adapt(cls, **kw)