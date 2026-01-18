from __future__ import annotations
import itertools
import re
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import clsregistry
from . import instrumentation
from . import interfaces
from . import mapperlib
from ._orm_constructors import composite
from ._orm_constructors import deferred
from ._orm_constructors import mapped_column
from ._orm_constructors import relationship
from ._orm_constructors import synonym
from .attributes import InstrumentedAttribute
from .base import _inspect_mapped_class
from .base import _is_mapped_class
from .base import Mapped
from .base import ORMDescriptor
from .decl_base import _add_attribute
from .decl_base import _as_declarative
from .decl_base import _ClassScanMapperConfig
from .decl_base import _declarative_constructor
from .decl_base import _DeferredMapperConfig
from .decl_base import _del_attribute
from .decl_base import _mapper
from .descriptor_props import Composite
from .descriptor_props import Synonym
from .descriptor_props import Synonym as _orm_synonym
from .mapper import Mapper
from .properties import MappedColumn
from .relationships import RelationshipProperty
from .state import InstanceState
from .. import exc
from .. import inspection
from .. import util
from ..sql import sqltypes
from ..sql.base import _NoArg
from ..sql.elements import SQLCoreOperations
from ..sql.schema import MetaData
from ..sql.selectable import FromClause
from ..util import hybridmethod
from ..util import hybridproperty
from ..util import typing as compat_typing
from ..util.typing import CallableReference
from ..util.typing import flatten_newtype
from ..util.typing import is_generic
from ..util.typing import is_literal
from ..util.typing import is_newtype
from ..util.typing import is_pep695
from ..util.typing import Literal
from ..util.typing import Self
class declared_attr(interfaces._MappedAttribute[_T], _declared_attr_common):
    """Mark a class-level method as representing the definition of
    a mapped property or Declarative directive.

    :class:`_orm.declared_attr` is typically applied as a decorator to a class
    level method, turning the attribute into a scalar-like property that can be
    invoked from the uninstantiated class. The Declarative mapping process
    looks for these :class:`_orm.declared_attr` callables as it scans classes,
    and assumes any attribute marked with :class:`_orm.declared_attr` will be a
    callable that will produce an object specific to the Declarative mapping or
    table configuration.

    :class:`_orm.declared_attr` is usually applicable to
    :ref:`mixins <orm_mixins_toplevel>`, to define relationships that are to be
    applied to different implementors of the class. It may also be used to
    define dynamically generated column expressions and other Declarative
    attributes.

    Example::

        class ProvidesUserMixin:
            "A mixin that adds a 'user' relationship to classes."

            user_id: Mapped[int] = mapped_column(ForeignKey("user_table.id"))

            @declared_attr
            def user(cls) -> Mapped["User"]:
                return relationship("User")

    When used with Declarative directives such as ``__tablename__``, the
    :meth:`_orm.declared_attr.directive` modifier may be used which indicates
    to :pep:`484` typing tools that the given method is not dealing with
    :class:`_orm.Mapped` attributes::

        class CreateTableName:
            @declared_attr.directive
            def __tablename__(cls) -> str:
                return cls.__name__.lower()

    :class:`_orm.declared_attr` can also be applied directly to mapped
    classes, to allow for attributes that dynamically configure themselves
    on subclasses when using mapped inheritance schemes.   Below
    illustrates :class:`_orm.declared_attr` to create a dynamic scheme
    for generating the :paramref:`_orm.Mapper.polymorphic_identity` parameter
    for subclasses::

        class Employee(Base):
            __tablename__ = 'employee'

            id: Mapped[int] = mapped_column(primary_key=True)
            type: Mapped[str] = mapped_column(String(50))

            @declared_attr.directive
            def __mapper_args__(cls) -> Dict[str, Any]:
                if cls.__name__ == 'Employee':
                    return {
                            "polymorphic_on":cls.type,
                            "polymorphic_identity":"Employee"
                    }
                else:
                    return {"polymorphic_identity":cls.__name__}

        class Engineer(Employee):
            pass

    :class:`_orm.declared_attr` supports decorating functions that are
    explicitly decorated with ``@classmethod``. This is never necessary from a
    runtime perspective, however may be needed in order to support :pep:`484`
    typing tools that don't otherwise recognize the decorated function as
    having class-level behaviors for the ``cls`` parameter::

        class SomethingMixin:
            x: Mapped[int]
            y: Mapped[int]

            @declared_attr
            @classmethod
            def x_plus_y(cls) -> Mapped[int]:
                return column_property(cls.x + cls.y)

    .. versionadded:: 2.0 - :class:`_orm.declared_attr` can accommodate a
       function decorated with ``@classmethod`` to help with :pep:`484`
       integration where needed.


    .. seealso::

        :ref:`orm_mixins_toplevel` - Declarative Mixin documentation with
        background on use patterns for :class:`_orm.declared_attr`.

    """
    if typing.TYPE_CHECKING:

        def __init__(self, fn: _DeclaredAttrDecorated[_T], cascading: bool=False):
            ...

        def __set__(self, instance: Any, value: Any) -> None:
            ...

        def __delete__(self, instance: Any) -> None:
            ...

        @overload
        def __get__(self, instance: None, owner: Any) -> InstrumentedAttribute[_T]:
            ...

        @overload
        def __get__(self, instance: object, owner: Any) -> _T:
            ...

        def __get__(self, instance: Optional[object], owner: Any) -> Union[InstrumentedAttribute[_T], _T]:
            ...

    @hybridmethod
    def _stateful(cls, **kw: Any) -> _stateful_declared_attr[_T]:
        return _stateful_declared_attr(**kw)

    @hybridproperty
    def directive(cls) -> _declared_directive[Any]:
        return _declared_directive

    @hybridproperty
    def cascading(cls) -> _stateful_declared_attr[_T]:
        return cls._stateful(cascading=True)