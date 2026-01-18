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
class registry:
    """Generalized registry for mapping classes.

    The :class:`_orm.registry` serves as the basis for maintaining a collection
    of mappings, and provides configurational hooks used to map classes.

    The three general kinds of mappings supported are Declarative Base,
    Declarative Decorator, and Imperative Mapping.   All of these mapping
    styles may be used interchangeably:

    * :meth:`_orm.registry.generate_base` returns a new declarative base
      class, and is the underlying implementation of the
      :func:`_orm.declarative_base` function.

    * :meth:`_orm.registry.mapped` provides a class decorator that will
      apply declarative mapping to a class without the use of a declarative
      base class.

    * :meth:`_orm.registry.map_imperatively` will produce a
      :class:`_orm.Mapper` for a class without scanning the class for
      declarative class attributes. This method suits the use case historically
      provided by the ``sqlalchemy.orm.mapper()`` classical mapping function,
      which is removed as of SQLAlchemy 2.0.

    .. versionadded:: 1.4

    .. seealso::

        :ref:`orm_mapping_classes_toplevel` - overview of class mapping
        styles.

    """
    _class_registry: clsregistry._ClsRegistryType
    _managers: weakref.WeakKeyDictionary[ClassManager[Any], Literal[True]]
    _non_primary_mappers: weakref.WeakKeyDictionary[Mapper[Any], Literal[True]]
    metadata: MetaData
    constructor: CallableReference[Callable[..., None]]
    type_annotation_map: _MutableTypeAnnotationMapType
    _dependents: Set[_RegistryType]
    _dependencies: Set[_RegistryType]
    _new_mappers: bool

    def __init__(self, *, metadata: Optional[MetaData]=None, class_registry: Optional[clsregistry._ClsRegistryType]=None, type_annotation_map: Optional[_TypeAnnotationMapType]=None, constructor: Callable[..., None]=_declarative_constructor):
        """Construct a new :class:`_orm.registry`

        :param metadata:
          An optional :class:`_schema.MetaData` instance.  All
          :class:`_schema.Table` objects generated using declarative
          table mapping will make use of this :class:`_schema.MetaData`
          collection.  If this argument is left at its default of ``None``,
          a blank :class:`_schema.MetaData` collection is created.

        :param constructor:
          Specify the implementation for the ``__init__`` function on a mapped
          class that has no ``__init__`` of its own.  Defaults to an
          implementation that assigns \\**kwargs for declared
          fields and relationships to an instance.  If ``None`` is supplied,
          no __init__ will be provided and construction will fall back to
          cls.__init__ by way of the normal Python semantics.

        :param class_registry: optional dictionary that will serve as the
          registry of class names-> mapped classes when string names
          are used to identify classes inside of :func:`_orm.relationship`
          and others.  Allows two or more declarative base classes
          to share the same registry of class names for simplified
          inter-base relationships.

        :param type_annotation_map: optional dictionary of Python types to
          SQLAlchemy :class:`_types.TypeEngine` classes or instances.
          The provided dict will update the default type mapping.  This
          is used exclusively by the :class:`_orm.MappedColumn` construct
          to produce column types based on annotations within the
          :class:`_orm.Mapped` type.

          .. versionadded:: 2.0

          .. seealso::

              :ref:`orm_declarative_mapped_column_type_map`


        """
        lcl_metadata = metadata or MetaData()
        if class_registry is None:
            class_registry = weakref.WeakValueDictionary()
        self._class_registry = class_registry
        self._managers = weakref.WeakKeyDictionary()
        self._non_primary_mappers = weakref.WeakKeyDictionary()
        self.metadata = lcl_metadata
        self.constructor = constructor
        self.type_annotation_map = {}
        if type_annotation_map is not None:
            self.update_type_annotation_map(type_annotation_map)
        self._dependents = set()
        self._dependencies = set()
        self._new_mappers = False
        with mapperlib._CONFIGURE_MUTEX:
            mapperlib._mapper_registries[self] = True

    def update_type_annotation_map(self, type_annotation_map: _TypeAnnotationMapType) -> None:
        """update the :paramref:`_orm.registry.type_annotation_map` with new
        values."""
        self.type_annotation_map.update({sub_type: sqltype for typ, sqltype in type_annotation_map.items() for sub_type in compat_typing.expand_unions(typ, include_union=True, discard_none=True)})

    def _resolve_type(self, python_type: _MatchedOnType) -> Optional[sqltypes.TypeEngine[Any]]:
        search: Iterable[Tuple[_MatchedOnType, Type[Any]]]
        python_type_type: Type[Any]
        if is_generic(python_type):
            if is_literal(python_type):
                python_type_type = cast('Type[Any]', python_type)
                search = ((python_type, python_type_type), (Literal, python_type_type))
            else:
                python_type_type = python_type.__origin__
                search = ((python_type, python_type_type),)
        elif is_newtype(python_type):
            python_type_type = flatten_newtype(python_type)
            search = ((python_type, python_type_type),)
        elif is_pep695(python_type):
            python_type_type = python_type.__value__
            flattened = None
            search = ((python_type, python_type_type),)
        else:
            python_type_type = cast('Type[Any]', python_type)
            flattened = None
            search = ((pt, pt) for pt in python_type_type.__mro__)
        for pt, flattened in search:
            sql_type = self.type_annotation_map.get(pt)
            if sql_type is None:
                sql_type = sqltypes._type_map_get(pt)
            if sql_type is not None:
                sql_type_inst = sqltypes.to_instance(sql_type)
                resolved_sql_type = sql_type_inst._resolve_for_python_type(python_type_type, pt, flattened)
                if resolved_sql_type is not None:
                    return resolved_sql_type
        return None

    @property
    def mappers(self) -> FrozenSet[Mapper[Any]]:
        """read only collection of all :class:`_orm.Mapper` objects."""
        return frozenset((manager.mapper for manager in self._managers)).union(self._non_primary_mappers)

    def _set_depends_on(self, registry: RegistryType) -> None:
        if registry is self:
            return
        registry._dependents.add(self)
        self._dependencies.add(registry)

    def _flag_new_mapper(self, mapper: Mapper[Any]) -> None:
        mapper._ready_for_configure = True
        if self._new_mappers:
            return
        for reg in self._recurse_with_dependents({self}):
            reg._new_mappers = True

    @classmethod
    def _recurse_with_dependents(cls, registries: Set[RegistryType]) -> Iterator[RegistryType]:
        todo = registries
        done = set()
        while todo:
            reg = todo.pop()
            done.add(reg)
            todo.update(reg._dependents.difference(done))
            yield reg
            todo.update(reg._dependents.difference(done))

    @classmethod
    def _recurse_with_dependencies(cls, registries: Set[RegistryType]) -> Iterator[RegistryType]:
        todo = registries
        done = set()
        while todo:
            reg = todo.pop()
            done.add(reg)
            todo.update(reg._dependencies.difference(done))
            yield reg
            todo.update(reg._dependencies.difference(done))

    def _mappers_to_configure(self) -> Iterator[Mapper[Any]]:
        return itertools.chain((manager.mapper for manager in list(self._managers) if manager.is_mapped and (not manager.mapper.configured) and manager.mapper._ready_for_configure), (npm for npm in list(self._non_primary_mappers) if not npm.configured and npm._ready_for_configure))

    def _add_non_primary_mapper(self, np_mapper: Mapper[Any]) -> None:
        self._non_primary_mappers[np_mapper] = True

    def _dispose_cls(self, cls: Type[_O]) -> None:
        clsregistry.remove_class(cls.__name__, cls, self._class_registry)

    def _add_manager(self, manager: ClassManager[Any]) -> None:
        self._managers[manager] = True
        if manager.is_mapped:
            raise exc.ArgumentError("Class '%s' already has a primary mapper defined. " % manager.class_)
        assert manager.registry is None
        manager.registry = self

    def configure(self, cascade: bool=False) -> None:
        """Configure all as-yet unconfigured mappers in this
        :class:`_orm.registry`.

        The configure step is used to reconcile and initialize the
        :func:`_orm.relationship` linkages between mapped classes, as well as
        to invoke configuration events such as the
        :meth:`_orm.MapperEvents.before_configured` and
        :meth:`_orm.MapperEvents.after_configured`, which may be used by ORM
        extensions or user-defined extension hooks.

        If one or more mappers in this registry contain
        :func:`_orm.relationship` constructs that refer to mapped classes in
        other registries, this registry is said to be *dependent* on those
        registries. In order to configure those dependent registries
        automatically, the :paramref:`_orm.registry.configure.cascade` flag
        should be set to ``True``. Otherwise, if they are not configured, an
        exception will be raised.  The rationale behind this behavior is to
        allow an application to programmatically invoke configuration of
        registries while controlling whether or not the process implicitly
        reaches other registries.

        As an alternative to invoking :meth:`_orm.registry.configure`, the ORM
        function :func:`_orm.configure_mappers` function may be used to ensure
        configuration is complete for all :class:`_orm.registry` objects in
        memory. This is generally simpler to use and also predates the usage of
        :class:`_orm.registry` objects overall. However, this function will
        impact all mappings throughout the running Python process and may be
        more memory/time consuming for an application that has many registries
        in use for different purposes that may not be needed immediately.

        .. seealso::

            :func:`_orm.configure_mappers`


        .. versionadded:: 1.4.0b2

        """
        mapperlib._configure_registries({self}, cascade=cascade)

    def dispose(self, cascade: bool=False) -> None:
        """Dispose of all mappers in this :class:`_orm.registry`.

        After invocation, all the classes that were mapped within this registry
        will no longer have class instrumentation associated with them. This
        method is the per-:class:`_orm.registry` analogue to the
        application-wide :func:`_orm.clear_mappers` function.

        If this registry contains mappers that are dependencies of other
        registries, typically via :func:`_orm.relationship` links, then those
        registries must be disposed as well. When such registries exist in
        relation to this one, their :meth:`_orm.registry.dispose` method will
        also be called, if the :paramref:`_orm.registry.dispose.cascade` flag
        is set to ``True``; otherwise, an error is raised if those registries
        were not already disposed.

        .. versionadded:: 1.4.0b2

        .. seealso::

            :func:`_orm.clear_mappers`

        """
        mapperlib._dispose_registries({self}, cascade=cascade)

    def _dispose_manager_and_mapper(self, manager: ClassManager[Any]) -> None:
        if 'mapper' in manager.__dict__:
            mapper = manager.mapper
            mapper._set_dispose_flags()
        class_ = manager.class_
        self._dispose_cls(class_)
        instrumentation._instrumentation_factory.unregister(class_)

    def generate_base(self, mapper: Optional[Callable[..., Mapper[Any]]]=None, cls: Type[Any]=object, name: str='Base', metaclass: Type[Any]=DeclarativeMeta) -> Any:
        """Generate a declarative base class.

        Classes that inherit from the returned class object will be
        automatically mapped using declarative mapping.

        E.g.::

            from sqlalchemy.orm import registry

            mapper_registry = registry()

            Base = mapper_registry.generate_base()

            class MyClass(Base):
                __tablename__ = "my_table"
                id = Column(Integer, primary_key=True)

        The above dynamically generated class is equivalent to the
        non-dynamic example below::

            from sqlalchemy.orm import registry
            from sqlalchemy.orm.decl_api import DeclarativeMeta

            mapper_registry = registry()

            class Base(metaclass=DeclarativeMeta):
                __abstract__ = True
                registry = mapper_registry
                metadata = mapper_registry.metadata

                __init__ = mapper_registry.constructor

        .. versionchanged:: 2.0 Note that the
           :meth:`_orm.registry.generate_base` method is superseded by the new
           :class:`_orm.DeclarativeBase` class, which generates a new "base"
           class using subclassing, rather than return value of a function.
           This allows an approach that is compatible with :pep:`484` typing
           tools.

        The :meth:`_orm.registry.generate_base` method provides the
        implementation for the :func:`_orm.declarative_base` function, which
        creates the :class:`_orm.registry` and base class all at once.

        See the section :ref:`orm_declarative_mapping` for background and
        examples.

        :param mapper:
          An optional callable, defaults to :class:`_orm.Mapper`.
          This function is used to generate new :class:`_orm.Mapper` objects.

        :param cls:
          Defaults to :class:`object`. A type to use as the base for the
          generated declarative base class. May be a class or tuple of classes.

        :param name:
          Defaults to ``Base``.  The display name for the generated
          class.  Customizing this is not required, but can improve clarity in
          tracebacks and debugging.

        :param metaclass:
          Defaults to :class:`.DeclarativeMeta`.  A metaclass or __metaclass__
          compatible callable to use as the meta type of the generated
          declarative base class.

        .. seealso::

            :ref:`orm_declarative_mapping`

            :func:`_orm.declarative_base`

        """
        metadata = self.metadata
        bases = not isinstance(cls, tuple) and (cls,) or cls
        class_dict: Dict[str, Any] = dict(registry=self, metadata=metadata)
        if isinstance(cls, type):
            class_dict['__doc__'] = cls.__doc__
        if self.constructor is not None:
            class_dict['__init__'] = self.constructor
        class_dict['__abstract__'] = True
        if mapper:
            class_dict['__mapper_cls__'] = mapper
        if hasattr(cls, '__class_getitem__'):

            def __class_getitem__(cls: Type[_T], key: Any) -> Type[_T]:
                return cls
            class_dict['__class_getitem__'] = __class_getitem__
        return metaclass(name, bases, class_dict)

    @compat_typing.dataclass_transform(field_specifiers=(MappedColumn, RelationshipProperty, Composite, Synonym, mapped_column, relationship, composite, synonym, deferred))
    @overload
    def mapped_as_dataclass(self, __cls: Type[_O]) -> Type[_O]:
        ...

    @overload
    def mapped_as_dataclass(self, __cls: Literal[None]=..., *, init: Union[_NoArg, bool]=..., repr: Union[_NoArg, bool]=..., eq: Union[_NoArg, bool]=..., order: Union[_NoArg, bool]=..., unsafe_hash: Union[_NoArg, bool]=..., match_args: Union[_NoArg, bool]=..., kw_only: Union[_NoArg, bool]=..., dataclass_callable: Union[_NoArg, Callable[..., Type[Any]]]=...) -> Callable[[Type[_O]], Type[_O]]:
        ...

    def mapped_as_dataclass(self, __cls: Optional[Type[_O]]=None, *, init: Union[_NoArg, bool]=_NoArg.NO_ARG, repr: Union[_NoArg, bool]=_NoArg.NO_ARG, eq: Union[_NoArg, bool]=_NoArg.NO_ARG, order: Union[_NoArg, bool]=_NoArg.NO_ARG, unsafe_hash: Union[_NoArg, bool]=_NoArg.NO_ARG, match_args: Union[_NoArg, bool]=_NoArg.NO_ARG, kw_only: Union[_NoArg, bool]=_NoArg.NO_ARG, dataclass_callable: Union[_NoArg, Callable[..., Type[Any]]]=_NoArg.NO_ARG) -> Union[Type[_O], Callable[[Type[_O]], Type[_O]]]:
        """Class decorator that will apply the Declarative mapping process
        to a given class, and additionally convert the class to be a
        Python dataclass.

        .. seealso::

            :ref:`orm_declarative_native_dataclasses` - complete background
            on SQLAlchemy native dataclass mapping


        .. versionadded:: 2.0


        """

        def decorate(cls: Type[_O]) -> Type[_O]:
            setattr(cls, '_sa_apply_dc_transforms', {'init': init, 'repr': repr, 'eq': eq, 'order': order, 'unsafe_hash': unsafe_hash, 'match_args': match_args, 'kw_only': kw_only, 'dataclass_callable': dataclass_callable})
            _as_declarative(self, cls, cls.__dict__)
            return cls
        if __cls:
            return decorate(__cls)
        else:
            return decorate

    def mapped(self, cls: Type[_O]) -> Type[_O]:
        """Class decorator that will apply the Declarative mapping process
        to a given class.

        E.g.::

            from sqlalchemy.orm import registry

            mapper_registry = registry()

            @mapper_registry.mapped
            class Foo:
                __tablename__ = 'some_table'

                id = Column(Integer, primary_key=True)
                name = Column(String)

        See the section :ref:`orm_declarative_mapping` for complete
        details and examples.

        :param cls: class to be mapped.

        :return: the class that was passed.

        .. seealso::

            :ref:`orm_declarative_mapping`

            :meth:`_orm.registry.generate_base` - generates a base class
            that will apply Declarative mapping to subclasses automatically
            using a Python metaclass.

        .. seealso::

            :meth:`_orm.registry.mapped_as_dataclass`

        """
        _as_declarative(self, cls, cls.__dict__)
        return cls

    def as_declarative_base(self, **kw: Any) -> Callable[[Type[_T]], Type[_T]]:
        """
        Class decorator which will invoke
        :meth:`_orm.registry.generate_base`
        for a given base class.

        E.g.::

            from sqlalchemy.orm import registry

            mapper_registry = registry()

            @mapper_registry.as_declarative_base()
            class Base:
                @declared_attr
                def __tablename__(cls):
                    return cls.__name__.lower()
                id = Column(Integer, primary_key=True)

            class MyMappedClass(Base):
                # ...

        All keyword arguments passed to
        :meth:`_orm.registry.as_declarative_base` are passed
        along to :meth:`_orm.registry.generate_base`.

        """

        def decorate(cls: Type[_T]) -> Type[_T]:
            kw['cls'] = cls
            kw['name'] = cls.__name__
            return self.generate_base(**kw)
        return decorate

    def map_declaratively(self, cls: Type[_O]) -> Mapper[_O]:
        """Map a class declaratively.

        In this form of mapping, the class is scanned for mapping information,
        including for columns to be associated with a table, and/or an
        actual table object.

        Returns the :class:`_orm.Mapper` object.

        E.g.::

            from sqlalchemy.orm import registry

            mapper_registry = registry()

            class Foo:
                __tablename__ = 'some_table'

                id = Column(Integer, primary_key=True)
                name = Column(String)

            mapper = mapper_registry.map_declaratively(Foo)

        This function is more conveniently invoked indirectly via either the
        :meth:`_orm.registry.mapped` class decorator or by subclassing a
        declarative metaclass generated from
        :meth:`_orm.registry.generate_base`.

        See the section :ref:`orm_declarative_mapping` for complete
        details and examples.

        :param cls: class to be mapped.

        :return: a :class:`_orm.Mapper` object.

        .. seealso::

            :ref:`orm_declarative_mapping`

            :meth:`_orm.registry.mapped` - more common decorator interface
            to this function.

            :meth:`_orm.registry.map_imperatively`

        """
        _as_declarative(self, cls, cls.__dict__)
        return cls.__mapper__

    def map_imperatively(self, class_: Type[_O], local_table: Optional[FromClause]=None, **kw: Any) -> Mapper[_O]:
        """Map a class imperatively.

        In this form of mapping, the class is not scanned for any mapping
        information.  Instead, all mapping constructs are passed as
        arguments.

        This method is intended to be fully equivalent to the now-removed
        SQLAlchemy ``mapper()`` function, except that it's in terms of
        a particular registry.

        E.g.::

            from sqlalchemy.orm import registry

            mapper_registry = registry()

            my_table = Table(
                "my_table",
                mapper_registry.metadata,
                Column('id', Integer, primary_key=True)
            )

            class MyClass:
                pass

            mapper_registry.map_imperatively(MyClass, my_table)

        See the section :ref:`orm_imperative_mapping` for complete background
        and usage examples.

        :param class\\_: The class to be mapped.  Corresponds to the
         :paramref:`_orm.Mapper.class_` parameter.

        :param local_table: the :class:`_schema.Table` or other
         :class:`_sql.FromClause` object that is the subject of the mapping.
         Corresponds to the
         :paramref:`_orm.Mapper.local_table` parameter.

        :param \\**kw: all other keyword arguments are passed to the
         :class:`_orm.Mapper` constructor directly.

        .. seealso::

            :ref:`orm_imperative_mapping`

            :ref:`orm_declarative_mapping`

        """
        return _mapper(self, class_, local_table, kw)