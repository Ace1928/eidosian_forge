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
class hybrid_method(interfaces.InspectionAttrInfo, Generic[_P, _R]):
    """A decorator which allows definition of a Python object method with both
    instance-level and class-level behavior.

    """
    is_attribute = True
    extension_type = HybridExtensionType.HYBRID_METHOD

    def __init__(self, func: Callable[Concatenate[Any, _P], _R], expr: Optional[Callable[Concatenate[Any, _P], SQLCoreOperations[_R]]]=None):
        """Create a new :class:`.hybrid_method`.

        Usage is typically via decorator::

            from sqlalchemy.ext.hybrid import hybrid_method

            class SomeClass:
                @hybrid_method
                def value(self, x, y):
                    return self._value + x + y

                @value.expression
                @classmethod
                def value(cls, x, y):
                    return func.some_function(cls._value, x, y)

        """
        self.func = func
        if expr is not None:
            self.expression(expr)
        else:
            self.expression(func)

    @property
    def inplace(self) -> Self:
        """Return the inplace mutator for this :class:`.hybrid_method`.

        The :class:`.hybrid_method` class already performs "in place" mutation
        when the :meth:`.hybrid_method.expression` decorator is called,
        so this attribute returns Self.

        .. versionadded:: 2.0.4

        .. seealso::

            :ref:`hybrid_pep484_naming`

        """
        return self

    @overload
    def __get__(self, instance: Literal[None], owner: Type[object]) -> Callable[_P, SQLCoreOperations[_R]]:
        ...

    @overload
    def __get__(self, instance: object, owner: Type[object]) -> Callable[_P, _R]:
        ...

    def __get__(self, instance: Optional[object], owner: Type[object]) -> Union[Callable[_P, _R], Callable[_P, SQLCoreOperations[_R]]]:
        if instance is None:
            return self.expr.__get__(owner, owner)
        else:
            return self.func.__get__(instance, owner)

    def expression(self, expr: Callable[Concatenate[Any, _P], SQLCoreOperations[_R]]) -> hybrid_method[_P, _R]:
        """Provide a modifying decorator that defines a
        SQL-expression producing method."""
        self.expr = expr
        if not self.expr.__doc__:
            self.expr.__doc__ = self.func.__doc__
        return self