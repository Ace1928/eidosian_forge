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
def _gen_dialect_impl(self, dialect: Dialect) -> TypeEngine[_T]:
    if dialect.name in self._variant_mapping:
        adapted = dialect.type_descriptor(self._variant_mapping[dialect.name])
    else:
        adapted = dialect.type_descriptor(self)
    if adapted is not self:
        return adapted
    typedesc = self.load_dialect_impl(dialect).dialect_impl(dialect)
    tt = self.copy()
    if not isinstance(tt, self.__class__):
        raise AssertionError('Type object %s does not properly implement the copy() method, it must return an object of type %s' % (self, self.__class__))
    tt.impl = tt.impl_instance = typedesc
    return tt