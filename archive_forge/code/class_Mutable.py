from within the mutable extension::
from __future__ import annotations
from collections import defaultdict
from typing import AbstractSet
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from weakref import WeakKeyDictionary
from .. import event
from .. import inspect
from .. import types
from .. import util
from ..orm import Mapper
from ..orm._typing import _ExternalEntityType
from ..orm._typing import _O
from ..orm._typing import _T
from ..orm.attributes import AttributeEventToken
from ..orm.attributes import flag_modified
from ..orm.attributes import InstrumentedAttribute
from ..orm.attributes import QueryableAttribute
from ..orm.context import QueryContext
from ..orm.decl_api import DeclarativeAttributeIntercept
from ..orm.state import InstanceState
from ..orm.unitofwork import UOWTransaction
from ..sql.base import SchemaEventTarget
from ..sql.schema import Column
from ..sql.type_api import TypeEngine
from ..util import memoized_property
from ..util.typing import SupportsIndex
from ..util.typing import TypeGuard
class Mutable(MutableBase):
    """Mixin that defines transparent propagation of change
    events to a parent object.

    See the example in :ref:`mutable_scalars` for usage information.

    """

    def changed(self) -> None:
        """Subclasses should call this method whenever change events occur."""
        for parent, key in self._parents.items():
            flag_modified(parent.obj(), key)

    @classmethod
    def associate_with_attribute(cls, attribute: InstrumentedAttribute[_O]) -> None:
        """Establish this type as a mutation listener for the given
        mapped descriptor.

        """
        cls._listen_on_attribute(attribute, True, attribute.class_)

    @classmethod
    def associate_with(cls, sqltype: type) -> None:
        """Associate this wrapper with all future mapped columns
        of the given type.

        This is a convenience method that calls
        ``associate_with_attribute`` automatically.

        .. warning::

           The listeners established by this method are *global*
           to all mappers, and are *not* garbage collected.   Only use
           :meth:`.associate_with` for types that are permanent to an
           application, not with ad-hoc types else this will cause unbounded
           growth in memory usage.

        """

        def listen_for_type(mapper: Mapper[_O], class_: type) -> None:
            if mapper.non_primary:
                return
            for prop in mapper.column_attrs:
                if isinstance(prop.columns[0].type, sqltype):
                    cls.associate_with_attribute(getattr(class_, prop.key))
        event.listen(Mapper, 'mapper_configured', listen_for_type)

    @classmethod
    def as_mutable(cls, sqltype: TypeEngine[_T]) -> TypeEngine[_T]:
        """Associate a SQL type with this mutable Python type.

        This establishes listeners that will detect ORM mappings against
        the given type, adding mutation event trackers to those mappings.

        The type is returned, unconditionally as an instance, so that
        :meth:`.as_mutable` can be used inline::

            Table('mytable', metadata,
                Column('id', Integer, primary_key=True),
                Column('data', MyMutableType.as_mutable(PickleType))
            )

        Note that the returned type is always an instance, even if a class
        is given, and that only columns which are declared specifically with
        that type instance receive additional instrumentation.

        To associate a particular mutable type with all occurrences of a
        particular type, use the :meth:`.Mutable.associate_with` classmethod
        of the particular :class:`.Mutable` subclass to establish a global
        association.

        .. warning::

           The listeners established by this method are *global*
           to all mappers, and are *not* garbage collected.   Only use
           :meth:`.as_mutable` for types that are permanent to an application,
           not with ad-hoc types else this will cause unbounded growth
           in memory usage.

        """
        sqltype = types.to_instance(sqltype)
        if isinstance(sqltype, SchemaEventTarget):

            @event.listens_for(sqltype, 'before_parent_attach')
            def _add_column_memo(sqltyp: TypeEngine[Any], parent: Column[_T]) -> None:
                parent.info['_ext_mutable_orig_type'] = sqltyp
            schema_event_check = True
        else:
            schema_event_check = False

        def listen_for_type(mapper: Mapper[_T], class_: Union[DeclarativeAttributeIntercept, type]) -> None:
            if mapper.non_primary:
                return
            _APPLIED_KEY = '_ext_mutable_listener_applied'
            for prop in mapper.column_attrs:
                if isinstance(prop.expression, Column) and (schema_event_check and prop.expression.info.get('_ext_mutable_orig_type') is sqltype or prop.expression.type is sqltype):
                    if not prop.expression.info.get(_APPLIED_KEY, False):
                        prop.expression.info[_APPLIED_KEY] = True
                        cls.associate_with_attribute(getattr(class_, prop.key))
        event.listen(Mapper, 'mapper_configured', listen_for_type)
        return sqltype