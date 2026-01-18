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
class DeclarativeBaseNoMeta(inspection.Inspectable[InstanceState[Any]]):
    """Same as :class:`_orm.DeclarativeBase`, but does not use a metaclass
    to intercept new attributes.

    The :class:`_orm.DeclarativeBaseNoMeta` base may be used when use of
    custom metaclasses is desirable.

    .. versionadded:: 2.0


    """
    _sa_registry: ClassVar[_RegistryType]
    registry: ClassVar[_RegistryType]
    'Refers to the :class:`_orm.registry` in use where new\n    :class:`_orm.Mapper` objects will be associated.'
    metadata: ClassVar[MetaData]
    'Refers to the :class:`_schema.MetaData` collection that will be used\n    for new :class:`_schema.Table` objects.\n\n    .. seealso::\n\n        :ref:`orm_declarative_metadata`\n\n    '
    __mapper__: ClassVar[Mapper[Any]]
    'The :class:`_orm.Mapper` object to which a particular class is\n    mapped.\n\n    May also be acquired using :func:`_sa.inspect`, e.g.\n    ``inspect(klass)``.\n\n    '
    __table__: Optional[FromClause]
    'The :class:`_sql.FromClause` to which a particular subclass is\n    mapped.\n\n    This is usually an instance of :class:`_schema.Table` but may also\n    refer to other kinds of :class:`_sql.FromClause` such as\n    :class:`_sql.Subquery`, depending on how the class is mapped.\n\n    .. seealso::\n\n        :ref:`orm_declarative_metadata`\n\n    '
    if typing.TYPE_CHECKING:

        def _sa_inspect_type(self) -> Mapper[Self]:
            ...

        def _sa_inspect_instance(self) -> InstanceState[Self]:
            ...
        __tablename__: Any
        'String name to assign to the generated\n        :class:`_schema.Table` object, if not specified directly via\n        :attr:`_orm.DeclarativeBase.__table__`.\n\n        .. seealso::\n\n            :ref:`orm_declarative_table`\n\n        '
        __mapper_args__: Any
        'Dictionary of arguments which will be passed to the\n        :class:`_orm.Mapper` constructor.\n\n        .. seealso::\n\n            :ref:`orm_declarative_mapper_options`\n\n        '
        __table_args__: Any
        'A dictionary or tuple of arguments that will be passed to the\n        :class:`_schema.Table` constructor.  See\n        :ref:`orm_declarative_table_configuration`\n        for background on the specific structure of this collection.\n\n        .. seealso::\n\n            :ref:`orm_declarative_table_configuration`\n\n        '

        def __init__(self, **kw: Any):
            ...

    def __init_subclass__(cls, **kw: Any) -> None:
        if DeclarativeBaseNoMeta in cls.__bases__:
            _check_not_declarative(cls, DeclarativeBaseNoMeta)
            _setup_declarative_base(cls)
        else:
            _as_declarative(cls._sa_registry, cls, cls.__dict__)
        super().__init_subclass__(**kw)