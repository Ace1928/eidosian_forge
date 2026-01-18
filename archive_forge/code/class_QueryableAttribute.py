from __future__ import annotations
import dataclasses
import operator
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import collections
from . import exc as orm_exc
from . import interfaces
from ._typing import insp_is_aliased_class
from .base import _DeclarativeMapped
from .base import ATTR_EMPTY
from .base import ATTR_WAS_SET
from .base import CALLABLES_OK
from .base import DEFERRED_HISTORY_LOAD
from .base import INCLUDE_PENDING_MUTATIONS  # noqa
from .base import INIT_OK
from .base import instance_dict as instance_dict
from .base import instance_state as instance_state
from .base import instance_str
from .base import LOAD_AGAINST_COMMITTED
from .base import LoaderCallableStatus
from .base import manager_of_class as manager_of_class
from .base import Mapped as Mapped  # noqa
from .base import NEVER_SET  # noqa
from .base import NO_AUTOFLUSH
from .base import NO_CHANGE  # noqa
from .base import NO_KEY
from .base import NO_RAISE
from .base import NO_VALUE
from .base import NON_PERSISTENT_OK  # noqa
from .base import opt_manager_of_class as opt_manager_of_class
from .base import PASSIVE_CLASS_MISMATCH  # noqa
from .base import PASSIVE_NO_FETCH
from .base import PASSIVE_NO_FETCH_RELATED  # noqa
from .base import PASSIVE_NO_INITIALIZE
from .base import PASSIVE_NO_RESULT
from .base import PASSIVE_OFF
from .base import PASSIVE_ONLY_PERSISTENT
from .base import PASSIVE_RETURN_NO_VALUE
from .base import PassiveFlag
from .base import RELATED_OBJECT_OK  # noqa
from .base import SQL_OK  # noqa
from .base import SQLORMExpression
from .base import state_str
from .. import event
from .. import exc
from .. import inspection
from .. import util
from ..event import dispatcher
from ..event import EventTarget
from ..sql import base as sql_base
from ..sql import cache_key
from ..sql import coercions
from ..sql import roles
from ..sql import visitors
from ..sql.cache_key import HasCacheKey
from ..sql.visitors import _TraverseInternalsType
from ..sql.visitors import InternalTraversal
from ..util.typing import Literal
from ..util.typing import Self
from ..util.typing import TypeGuard
@inspection._self_inspects
class QueryableAttribute(_DeclarativeMapped[_T_co], SQLORMExpression[_T_co], interfaces.InspectionAttr, interfaces.PropComparator[_T_co], roles.JoinTargetRole, roles.OnClauseRole, sql_base.Immutable, cache_key.SlotsMemoizedHasCacheKey, util.MemoizedSlots, EventTarget):
    """Base class for :term:`descriptor` objects that intercept
    attribute events on behalf of a :class:`.MapperProperty`
    object.  The actual :class:`.MapperProperty` is accessible
    via the :attr:`.QueryableAttribute.property`
    attribute.


    .. seealso::

        :class:`.InstrumentedAttribute`

        :class:`.MapperProperty`

        :attr:`_orm.Mapper.all_orm_descriptors`

        :attr:`_orm.Mapper.attrs`
    """
    __slots__ = ('class_', 'key', 'impl', 'comparator', 'property', 'parent', 'expression', '_of_type', '_extra_criteria', '_slots_dispatch', '_propagate_attrs', '_doc')
    is_attribute = True
    dispatch: dispatcher[QueryableAttribute[_T_co]]
    class_: _ExternalEntityType[Any]
    key: str
    parententity: _InternalEntityType[Any]
    impl: AttributeImpl
    comparator: interfaces.PropComparator[_T_co]
    _of_type: Optional[_InternalEntityType[Any]]
    _extra_criteria: Tuple[ColumnElement[bool], ...]
    _doc: Optional[str]
    __visit_name__ = 'orm_instrumented_attribute'

    def __init__(self, class_: _ExternalEntityType[_O], key: str, parententity: _InternalEntityType[_O], comparator: interfaces.PropComparator[_T_co], impl: Optional[AttributeImpl]=None, of_type: Optional[_InternalEntityType[Any]]=None, extra_criteria: Tuple[ColumnElement[bool], ...]=()):
        self.class_ = class_
        self.key = key
        self._parententity = self.parent = parententity
        self.impl = impl
        assert comparator is not None
        self.comparator = comparator
        self._of_type = of_type
        self._extra_criteria = extra_criteria
        self._doc = None
        manager = opt_manager_of_class(class_)
        if manager:
            for base in manager._bases:
                if key in base:
                    self.dispatch._update(base[key].dispatch)
                    if base[key].dispatch._active_history:
                        self.dispatch._active_history = True
    _cache_key_traversal = [('key', visitors.ExtendedInternalTraversal.dp_string), ('_parententity', visitors.ExtendedInternalTraversal.dp_multi), ('_of_type', visitors.ExtendedInternalTraversal.dp_multi), ('_extra_criteria', visitors.InternalTraversal.dp_clauseelement_list)]

    def __reduce__(self) -> Any:
        return (_queryable_attribute_unreduce, (self.key, self._parententity.mapper.class_, self._parententity, self._parententity.entity))

    @property
    def _impl_uses_objects(self) -> bool:
        return self.impl.uses_objects

    def get_history(self, instance: Any, passive: PassiveFlag=PASSIVE_OFF) -> History:
        return self.impl.get_history(instance_state(instance), instance_dict(instance), passive)

    @property
    def info(self) -> _InfoType:
        """Return the 'info' dictionary for the underlying SQL element.

        The behavior here is as follows:

        * If the attribute is a column-mapped property, i.e.
          :class:`.ColumnProperty`, which is mapped directly
          to a schema-level :class:`_schema.Column` object, this attribute
          will return the :attr:`.SchemaItem.info` dictionary associated
          with the core-level :class:`_schema.Column` object.

        * If the attribute is a :class:`.ColumnProperty` but is mapped to
          any other kind of SQL expression other than a
          :class:`_schema.Column`,
          the attribute will refer to the :attr:`.MapperProperty.info`
          dictionary associated directly with the :class:`.ColumnProperty`,
          assuming the SQL expression itself does not have its own ``.info``
          attribute (which should be the case, unless a user-defined SQL
          construct has defined one).

        * If the attribute refers to any other kind of
          :class:`.MapperProperty`, including :class:`.Relationship`,
          the attribute will refer to the :attr:`.MapperProperty.info`
          dictionary associated with that :class:`.MapperProperty`.

        * To access the :attr:`.MapperProperty.info` dictionary of the
          :class:`.MapperProperty` unconditionally, including for a
          :class:`.ColumnProperty` that's associated directly with a
          :class:`_schema.Column`, the attribute can be referred to using
          :attr:`.QueryableAttribute.property` attribute, as
          ``MyClass.someattribute.property.info``.

        .. seealso::

            :attr:`.SchemaItem.info`

            :attr:`.MapperProperty.info`

        """
        return self.comparator.info
    parent: _InternalEntityType[Any]
    'Return an inspection instance representing the parent.\n\n    This will be either an instance of :class:`_orm.Mapper`\n    or :class:`.AliasedInsp`, depending upon the nature\n    of the parent entity which this attribute is associated\n    with.\n\n    '
    expression: ColumnElement[_T_co]
    'The SQL expression object represented by this\n    :class:`.QueryableAttribute`.\n\n    This will typically be an instance of a :class:`_sql.ColumnElement`\n    subclass representing a column expression.\n\n    '

    def _memoized_attr_expression(self) -> ColumnElement[_T]:
        annotations: _AnnotationDict
        entity_namespace = self._entity_namespace
        assert isinstance(entity_namespace, HasCacheKey)
        if self.key is _UNKNOWN_ATTR_KEY:
            annotations = {'entity_namespace': entity_namespace}
        else:
            annotations = {'proxy_key': self.key, 'proxy_owner': self._parententity, 'entity_namespace': entity_namespace}
        ce = self.comparator.__clause_element__()
        try:
            if TYPE_CHECKING:
                assert isinstance(ce, ColumnElement)
            anno = ce._annotate
        except AttributeError as ae:
            raise exc.InvalidRequestError('When interpreting attribute "%s" as a SQL expression, expected __clause_element__() to return a ClauseElement object, got: %r' % (self, ce)) from ae
        else:
            return anno(annotations)

    def _memoized_attr__propagate_attrs(self) -> _PropagateAttrsType:
        return util.immutabledict({'compile_state_plugin': 'orm', 'plugin_subject': self._parentmapper})

    @property
    def _entity_namespace(self) -> _InternalEntityType[Any]:
        return self._parententity

    @property
    def _annotations(self) -> _AnnotationDict:
        return self.__clause_element__()._annotations

    def __clause_element__(self) -> ColumnElement[_T_co]:
        return self.expression

    @property
    def _from_objects(self) -> List[FromClause]:
        return self.expression._from_objects

    def _bulk_update_tuples(self, value: Any) -> Sequence[Tuple[_DMLColumnArgument, Any]]:
        """Return setter tuples for a bulk UPDATE."""
        return self.comparator._bulk_update_tuples(value)

    def adapt_to_entity(self, adapt_to_entity: AliasedInsp[Any]) -> Self:
        assert not self._of_type
        return self.__class__(adapt_to_entity.entity, self.key, impl=self.impl, comparator=self.comparator.adapt_to_entity(adapt_to_entity), parententity=adapt_to_entity)

    def of_type(self, entity: _EntityType[Any]) -> QueryableAttribute[_T]:
        return QueryableAttribute(self.class_, self.key, self._parententity, impl=self.impl, comparator=self.comparator.of_type(entity), of_type=inspection.inspect(entity), extra_criteria=self._extra_criteria)

    def and_(self, *clauses: _ColumnExpressionArgument[bool]) -> QueryableAttribute[bool]:
        if TYPE_CHECKING:
            assert isinstance(self.comparator, RelationshipProperty.Comparator)
        exprs = tuple((coercions.expect(roles.WhereHavingRole, clause) for clause in util.coerce_generator_arg(clauses)))
        return QueryableAttribute(self.class_, self.key, self._parententity, impl=self.impl, comparator=self.comparator.and_(*exprs), of_type=self._of_type, extra_criteria=self._extra_criteria + exprs)

    def _clone(self, **kw: Any) -> QueryableAttribute[_T]:
        return QueryableAttribute(self.class_, self.key, self._parententity, impl=self.impl, comparator=self.comparator, of_type=self._of_type, extra_criteria=self._extra_criteria)

    def label(self, name: Optional[str]) -> Label[_T_co]:
        return self.__clause_element__().label(name)

    def operate(self, op: OperatorType, *other: Any, **kwargs: Any) -> ColumnElement[Any]:
        return op(self.comparator, *other, **kwargs)

    def reverse_operate(self, op: OperatorType, other: Any, **kwargs: Any) -> ColumnElement[Any]:
        return op(other, self.comparator, **kwargs)

    def hasparent(self, state: InstanceState[Any], optimistic: bool=False) -> bool:
        return self.impl.hasparent(state, optimistic=optimistic) is not False

    def __getattr__(self, key: str) -> Any:
        try:
            return util.MemoizedSlots.__getattr__(self, key)
        except AttributeError:
            pass
        try:
            return getattr(self.comparator, key)
        except AttributeError as err:
            raise AttributeError('Neither %r object nor %r object associated with %s has an attribute %r' % (type(self).__name__, type(self.comparator).__name__, self, key)) from err

    def __str__(self) -> str:
        return f'{self.class_.__name__}.{self.key}'

    def _memoized_attr_property(self) -> Optional[MapperProperty[Any]]:
        return self.comparator.property