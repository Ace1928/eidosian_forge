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
class PropComparator(SQLORMOperations[_T_co], Generic[_T_co], ColumnOperators):
    """Defines SQL operations for ORM mapped attributes.

    SQLAlchemy allows for operators to
    be redefined at both the Core and ORM level.  :class:`.PropComparator`
    is the base class of operator redefinition for ORM-level operations,
    including those of :class:`.ColumnProperty`,
    :class:`.Relationship`, and :class:`.Composite`.

    User-defined subclasses of :class:`.PropComparator` may be created. The
    built-in Python comparison and math operator methods, such as
    :meth:`.operators.ColumnOperators.__eq__`,
    :meth:`.operators.ColumnOperators.__lt__`, and
    :meth:`.operators.ColumnOperators.__add__`, can be overridden to provide
    new operator behavior. The custom :class:`.PropComparator` is passed to
    the :class:`.MapperProperty` instance via the ``comparator_factory``
    argument. In each case,
    the appropriate subclass of :class:`.PropComparator` should be used::

        # definition of custom PropComparator subclasses

        from sqlalchemy.orm.properties import \\
                                ColumnProperty,\\
                                Composite,\\
                                Relationship

        class MyColumnComparator(ColumnProperty.Comparator):
            def __eq__(self, other):
                return self.__clause_element__() == other

        class MyRelationshipComparator(Relationship.Comparator):
            def any(self, expression):
                "define the 'any' operation"
                # ...

        class MyCompositeComparator(Composite.Comparator):
            def __gt__(self, other):
                "redefine the 'greater than' operation"

                return sql.and_(*[a>b for a, b in
                                  zip(self.__clause_element__().clauses,
                                      other.__composite_values__())])


        # application of custom PropComparator subclasses

        from sqlalchemy.orm import column_property, relationship, composite
        from sqlalchemy import Column, String

        class SomeMappedClass(Base):
            some_column = column_property(Column("some_column", String),
                                comparator_factory=MyColumnComparator)

            some_relationship = relationship(SomeOtherClass,
                                comparator_factory=MyRelationshipComparator)

            some_composite = composite(
                    Column("a", String), Column("b", String),
                    comparator_factory=MyCompositeComparator
                )

    Note that for column-level operator redefinition, it's usually
    simpler to define the operators at the Core level, using the
    :attr:`.TypeEngine.comparator_factory` attribute.  See
    :ref:`types_operators` for more detail.

    .. seealso::

        :class:`.ColumnProperty.Comparator`

        :class:`.Relationship.Comparator`

        :class:`.Composite.Comparator`

        :class:`.ColumnOperators`

        :ref:`types_operators`

        :attr:`.TypeEngine.comparator_factory`

    """
    __slots__ = ('prop', '_parententity', '_adapt_to_entity')
    __visit_name__ = 'orm_prop_comparator'
    _parententity: _InternalEntityType[Any]
    _adapt_to_entity: Optional[AliasedInsp[Any]]
    prop: RODescriptorReference[MapperProperty[_T_co]]

    def __init__(self, prop: MapperProperty[_T], parentmapper: _InternalEntityType[Any], adapt_to_entity: Optional[AliasedInsp[Any]]=None):
        self.prop = prop
        self._parententity = adapt_to_entity or parentmapper
        self._adapt_to_entity = adapt_to_entity

    @util.non_memoized_property
    def property(self) -> MapperProperty[_T_co]:
        """Return the :class:`.MapperProperty` associated with this
        :class:`.PropComparator`.


        Return values here will commonly be instances of
        :class:`.ColumnProperty` or :class:`.Relationship`.


        """
        return self.prop

    def __clause_element__(self) -> roles.ColumnsClauseRole:
        raise NotImplementedError('%r' % self)

    def _bulk_update_tuples(self, value: Any) -> Sequence[Tuple[_DMLColumnArgument, Any]]:
        """Receive a SQL expression that represents a value in the SET
        clause of an UPDATE statement.

        Return a tuple that can be passed to a :class:`_expression.Update`
        construct.

        """
        return [(cast('_DMLColumnArgument', self.__clause_element__()), value)]

    def adapt_to_entity(self, adapt_to_entity: AliasedInsp[Any]) -> PropComparator[_T_co]:
        """Return a copy of this PropComparator which will use the given
        :class:`.AliasedInsp` to produce corresponding expressions.
        """
        return self.__class__(self.prop, self._parententity, adapt_to_entity)

    @util.ro_non_memoized_property
    def _parentmapper(self) -> Mapper[Any]:
        """legacy; this is renamed to _parententity to be
        compatible with QueryableAttribute."""
        return self._parententity.mapper

    def _criterion_exists(self, criterion: Optional[_ColumnExpressionArgument[bool]]=None, **kwargs: Any) -> ColumnElement[Any]:
        return self.prop.comparator._criterion_exists(criterion, **kwargs)

    @util.ro_non_memoized_property
    def adapter(self) -> Optional[_ORMAdapterProto]:
        """Produce a callable that adapts column expressions
        to suit an aliased version of this comparator.

        """
        if self._adapt_to_entity is None:
            return None
        else:
            return self._adapt_to_entity._orm_adapt_element

    @util.ro_non_memoized_property
    def info(self) -> _InfoType:
        return self.prop.info

    @staticmethod
    def _any_op(a: Any, b: Any, **kwargs: Any) -> Any:
        return a.any(b, **kwargs)

    @staticmethod
    def _has_op(left: Any, other: Any, **kwargs: Any) -> Any:
        return left.has(other, **kwargs)

    @staticmethod
    def _of_type_op(a: Any, class_: Any) -> Any:
        return a.of_type(class_)
    any_op = cast(operators.OperatorType, _any_op)
    has_op = cast(operators.OperatorType, _has_op)
    of_type_op = cast(operators.OperatorType, _of_type_op)
    if typing.TYPE_CHECKING:

        def operate(self, op: OperatorType, *other: Any, **kwargs: Any) -> ColumnElement[Any]:
            ...

        def reverse_operate(self, op: OperatorType, other: Any, **kwargs: Any) -> ColumnElement[Any]:
            ...

    def of_type(self, class_: _EntityType[Any]) -> PropComparator[_T_co]:
        """Redefine this object in terms of a polymorphic subclass,
        :func:`_orm.with_polymorphic` construct, or :func:`_orm.aliased`
        construct.

        Returns a new PropComparator from which further criterion can be
        evaluated.

        e.g.::

            query.join(Company.employees.of_type(Engineer)).\\
               filter(Engineer.name=='foo')

        :param \\class_: a class or mapper indicating that criterion will be
            against this specific subclass.

        .. seealso::

            :ref:`orm_queryguide_joining_relationships_aliased` - in the
            :ref:`queryguide_toplevel`

            :ref:`inheritance_of_type`

        """
        return self.operate(PropComparator.of_type_op, class_)

    def and_(self, *criteria: _ColumnExpressionArgument[bool]) -> PropComparator[bool]:
        """Add additional criteria to the ON clause that's represented by this
        relationship attribute.

        E.g.::


            stmt = select(User).join(
                User.addresses.and_(Address.email_address != 'foo')
            )

            stmt = select(User).options(
                joinedload(User.addresses.and_(Address.email_address != 'foo'))
            )

        .. versionadded:: 1.4

        .. seealso::

            :ref:`orm_queryguide_join_on_augmented`

            :ref:`loader_option_criteria`

            :func:`.with_loader_criteria`

        """
        return self.operate(operators.and_, *criteria)

    def any(self, criterion: Optional[_ColumnExpressionArgument[bool]]=None, **kwargs: Any) -> ColumnElement[bool]:
        """Return a SQL expression representing true if this element
        references a member which meets the given criterion.

        The usual implementation of ``any()`` is
        :meth:`.Relationship.Comparator.any`.

        :param criterion: an optional ClauseElement formulated against the
          member class' table or attributes.

        :param \\**kwargs: key/value pairs corresponding to member class
          attribute names which will be compared via equality to the
          corresponding values.

        """
        return self.operate(PropComparator.any_op, criterion, **kwargs)

    def has(self, criterion: Optional[_ColumnExpressionArgument[bool]]=None, **kwargs: Any) -> ColumnElement[bool]:
        """Return a SQL expression representing true if this element
        references a member which meets the given criterion.

        The usual implementation of ``has()`` is
        :meth:`.Relationship.Comparator.has`.

        :param criterion: an optional ClauseElement formulated against the
          member class' table or attributes.

        :param \\**kwargs: key/value pairs corresponding to member class
          attribute names which will be compared via equality to the
          corresponding values.

        """
        return self.operate(PropComparator.has_op, criterion, **kwargs)