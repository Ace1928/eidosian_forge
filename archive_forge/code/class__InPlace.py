from __future__ import annotations
from typing import Any
from typing import Callable
from typing import cast
from typing import Generic
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import util
from ..orm import attributes
from ..orm import InspectionAttrExtensionType
from ..orm import interfaces
from ..orm import ORMDescriptor
from ..orm.attributes import QueryableAttribute
from ..sql import roles
from ..sql._typing import is_has_clause_element
from ..sql.elements import ColumnElement
from ..sql.elements import SQLCoreOperations
from ..util.typing import Concatenate
from ..util.typing import Literal
from ..util.typing import ParamSpec
from ..util.typing import Protocol
from ..util.typing import Self
class _InPlace(Generic[_TE]):
    """A builder helper for .hybrid_property.

        .. versionadded:: 2.0.4

        """
    __slots__ = ('attr',)

    def __init__(self, attr: hybrid_property[_TE]):
        self.attr = attr

    def _set(self, **kw: Any) -> hybrid_property[_TE]:
        for k, v in kw.items():
            setattr(self.attr, k, _unwrap_classmethod(v))
        return self.attr

    def getter(self, fget: _HybridGetterType[_TE]) -> hybrid_property[_TE]:
        return self._set(fget=fget)

    def setter(self, fset: _HybridSetterType[_TE]) -> hybrid_property[_TE]:
        return self._set(fset=fset)

    def deleter(self, fdel: _HybridDeleterType[_TE]) -> hybrid_property[_TE]:
        return self._set(fdel=fdel)

    def expression(self, expr: _HybridExprCallableType[_TE]) -> hybrid_property[_TE]:
        return self._set(expr=expr)

    def comparator(self, comparator: _HybridComparatorCallableType[_TE]) -> hybrid_property[_TE]:
        return self._set(custom_comparator=comparator)

    def update_expression(self, meth: _HybridUpdaterType[_TE]) -> hybrid_property[_TE]:
        return self._set(update_expr=meth)