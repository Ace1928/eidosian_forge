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
@util.ro_memoized_property
def _generic_type_affinity(self) -> Type[TypeEngine[_T]]:
    best_camelcase = None
    best_uppercase = None
    if not isinstance(self, TypeEngine):
        return self.__class__
    for t in self.__class__.__mro__:
        if t.__module__ in ('sqlalchemy.sql.sqltypes', 'sqlalchemy.sql.type_api') and issubclass(t, TypeEngine) and (TypeEngineMixin not in t.__bases__) and (t not in (TypeEngine, TypeEngineMixin)) and (t.__name__[0] != '_'):
            if t.__name__.isupper() and (not best_uppercase):
                best_uppercase = t
            elif not t.__name__.isupper() and (not best_camelcase):
                best_camelcase = t
    return best_camelcase or best_uppercase or cast('Type[TypeEngine[_T]]', NULLTYPE.__class__)