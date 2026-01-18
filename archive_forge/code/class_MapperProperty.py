from __future__ import annotations
import collections
import dataclasses
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Generic
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import exc as orm_exc
from . import path_registry
from .base import _MappedAttribute as _MappedAttribute
from .base import EXT_CONTINUE as EXT_CONTINUE  # noqa: F401
from .base import EXT_SKIP as EXT_SKIP  # noqa: F401
from .base import EXT_STOP as EXT_STOP  # noqa: F401
from .base import InspectionAttr as InspectionAttr  # noqa: F401
from .base import InspectionAttrInfo as InspectionAttrInfo
from .base import MANYTOMANY as MANYTOMANY  # noqa: F401
from .base import MANYTOONE as MANYTOONE  # noqa: F401
from .base import NO_KEY as NO_KEY  # noqa: F401
from .base import NO_VALUE as NO_VALUE  # noqa: F401
from .base import NotExtension as NotExtension  # noqa: F401
from .base import ONETOMANY as ONETOMANY  # noqa: F401
from .base import RelationshipDirection as RelationshipDirection  # noqa: F401
from .base import SQLORMOperations
from .. import ColumnElement
from .. import exc as sa_exc
from .. import inspection
from .. import util
from ..sql import operators
from ..sql import roles
from ..sql import visitors
from ..sql.base import _NoArg
from ..sql.base import ExecutableOption
from ..sql.cache_key import HasCacheKey
from ..sql.operators import ColumnOperators
from ..sql.schema import Column
from ..sql.type_api import TypeEngine
from ..util import warn_deprecated
from ..util.typing import RODescriptorReference
from ..util.typing import TypedDict
@inspection._self_inspects
class MapperProperty(HasCacheKey, _DCAttributeOptions, _MappedAttribute[_T], InspectionAttrInfo, util.MemoizedSlots):
    """Represent a particular class attribute mapped by :class:`_orm.Mapper`.

    The most common occurrences of :class:`.MapperProperty` are the
    mapped :class:`_schema.Column`, which is represented in a mapping as
    an instance of :class:`.ColumnProperty`,
    and a reference to another class produced by :func:`_orm.relationship`,
    represented in the mapping as an instance of
    :class:`.Relationship`.

    """
    __slots__ = ('_configure_started', '_configure_finished', '_attribute_options', '_has_dataclass_arguments', 'parent', 'key', 'info', 'doc')
    _cache_key_traversal: _TraverseInternalsType = [('parent', visitors.ExtendedInternalTraversal.dp_has_cache_key), ('key', visitors.ExtendedInternalTraversal.dp_string)]
    if not TYPE_CHECKING:
        cascade = None
    is_property = True
    'Part of the InspectionAttr interface; states this object is a\n    mapper property.\n\n    '
    comparator: PropComparator[_T]
    'The :class:`_orm.PropComparator` instance that implements SQL\n    expression construction on behalf of this mapped attribute.'
    key: str
    'name of class attribute'
    parent: Mapper[Any]
    'the :class:`.Mapper` managing this property.'
    _is_relationship = False
    _links_to_entity: bool
    'True if this MapperProperty refers to a mapped entity.\n\n    Should only be True for Relationship, False for all others.\n\n    '
    doc: Optional[str]
    'optional documentation string'
    info: _InfoType
    'Info dictionary associated with the object, allowing user-defined\n    data to be associated with this :class:`.InspectionAttr`.\n\n    The dictionary is generated when first accessed.  Alternatively,\n    it can be specified as a constructor argument to the\n    :func:`.column_property`, :func:`_orm.relationship`, or :func:`.composite`\n    functions.\n\n    .. seealso::\n\n        :attr:`.QueryableAttribute.info`\n\n        :attr:`.SchemaItem.info`\n\n    '

    def _memoized_attr_info(self) -> _InfoType:
        """Info dictionary associated with the object, allowing user-defined
        data to be associated with this :class:`.InspectionAttr`.

        The dictionary is generated when first accessed.  Alternatively,
        it can be specified as a constructor argument to the
        :func:`.column_property`, :func:`_orm.relationship`, or
        :func:`.composite`
        functions.

        .. seealso::

            :attr:`.QueryableAttribute.info`

            :attr:`.SchemaItem.info`

        """
        return {}

    def setup(self, context: ORMCompileState, query_entity: _MapperEntity, path: AbstractEntityRegistry, adapter: Optional[ORMAdapter], **kwargs: Any) -> None:
        """Called by Query for the purposes of constructing a SQL statement.

        Each MapperProperty associated with the target mapper processes the
        statement referenced by the query context, adding columns and/or
        criterion as appropriate.

        """

    def create_row_processor(self, context: ORMCompileState, query_entity: _MapperEntity, path: AbstractEntityRegistry, mapper: Mapper[Any], result: Result[Any], adapter: Optional[ORMAdapter], populators: _PopulatorDict) -> None:
        """Produce row processing functions and append to the given
        set of populators lists.

        """

    def cascade_iterator(self, type_: str, state: InstanceState[Any], dict_: _InstanceDict, visited_states: Set[InstanceState[Any]], halt_on: Optional[Callable[[InstanceState[Any]], bool]]=None) -> Iterator[Tuple[object, Mapper[Any], InstanceState[Any], _InstanceDict]]:
        """Iterate through instances related to the given instance for
        a particular 'cascade', starting with this MapperProperty.

        Return an iterator3-tuples (instance, mapper, state).

        Note that the 'cascade' collection on this MapperProperty is
        checked first for the given type before cascade_iterator is called.

        This method typically only applies to Relationship.

        """
        return iter(())

    def set_parent(self, parent: Mapper[Any], init: bool) -> None:
        """Set the parent mapper that references this MapperProperty.

        This method is overridden by some subclasses to perform extra
        setup when the mapper is first known.

        """
        self.parent = parent

    def instrument_class(self, mapper: Mapper[Any]) -> None:
        """Hook called by the Mapper to the property to initiate
        instrumentation of the class attribute managed by this
        MapperProperty.

        The MapperProperty here will typically call out to the
        attributes module to set up an InstrumentedAttribute.

        This step is the first of two steps to set up an InstrumentedAttribute,
        and is called early in the mapper setup process.

        The second step is typically the init_class_attribute step,
        called from StrategizedProperty via the post_instrument_class()
        hook.  This step assigns additional state to the InstrumentedAttribute
        (specifically the "impl") which has been determined after the
        MapperProperty has determined what kind of persistence
        management it needs to do (e.g. scalar, object, collection, etc).

        """

    def __init__(self, attribute_options: Optional[_AttributeOptions]=None, _assume_readonly_dc_attributes: bool=False) -> None:
        self._configure_started = False
        self._configure_finished = False
        if _assume_readonly_dc_attributes:
            default_attrs = _DEFAULT_READONLY_ATTRIBUTE_OPTIONS
        else:
            default_attrs = _DEFAULT_ATTRIBUTE_OPTIONS
        if attribute_options and attribute_options != default_attrs:
            self._has_dataclass_arguments = True
            self._attribute_options = attribute_options
        else:
            self._has_dataclass_arguments = False
            self._attribute_options = default_attrs

    def init(self) -> None:
        """Called after all mappers are created to assemble
        relationships between mappers and perform other post-mapper-creation
        initialization steps.


        """
        self._configure_started = True
        self.do_init()
        self._configure_finished = True

    @property
    def class_attribute(self) -> InstrumentedAttribute[_T]:
        """Return the class-bound descriptor corresponding to this
        :class:`.MapperProperty`.

        This is basically a ``getattr()`` call::

            return getattr(self.parent.class_, self.key)

        I.e. if this :class:`.MapperProperty` were named ``addresses``,
        and the class to which it is mapped is ``User``, this sequence
        is possible::

            >>> from sqlalchemy import inspect
            >>> mapper = inspect(User)
            >>> addresses_property = mapper.attrs.addresses
            >>> addresses_property.class_attribute is User.addresses
            True
            >>> User.addresses.property is addresses_property
            True


        """
        return getattr(self.parent.class_, self.key)

    def do_init(self) -> None:
        """Perform subclass-specific initialization post-mapper-creation
        steps.

        This is a template method called by the ``MapperProperty``
        object's init() method.

        """

    def post_instrument_class(self, mapper: Mapper[Any]) -> None:
        """Perform instrumentation adjustments that need to occur
        after init() has completed.

        The given Mapper is the Mapper invoking the operation, which
        may not be the same Mapper as self.parent in an inheritance
        scenario; however, Mapper will always at least be a sub-mapper of
        self.parent.

        This method is typically used by StrategizedProperty, which delegates
        it to LoaderStrategy.init_class_attribute() to perform final setup
        on the class-bound InstrumentedAttribute.

        """

    def merge(self, session: Session, source_state: InstanceState[Any], source_dict: _InstanceDict, dest_state: InstanceState[Any], dest_dict: _InstanceDict, load: bool, _recursive: Dict[Any, object], _resolve_conflict_map: Dict[_IdentityKeyType[Any], object]) -> None:
        """Merge the attribute represented by this ``MapperProperty``
        from source to destination object.

        """

    def __repr__(self) -> str:
        return '<%s at 0x%x; %s>' % (self.__class__.__name__, id(self), getattr(self, 'key', 'no key'))