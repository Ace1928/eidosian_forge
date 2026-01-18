from __future__ import annotations
import collections
from collections import abc
import dataclasses
import inspect as _py_inspect
import itertools
import re
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import strategy_options
from ._typing import insp_is_aliased_class
from ._typing import is_has_collection_adapter
from .base import _DeclarativeMapped
from .base import _is_mapped_class
from .base import class_mapper
from .base import DynamicMapped
from .base import LoaderCallableStatus
from .base import PassiveFlag
from .base import state_str
from .base import WriteOnlyMapped
from .interfaces import _AttributeOptions
from .interfaces import _IntrospectsAnnotations
from .interfaces import MANYTOMANY
from .interfaces import MANYTOONE
from .interfaces import ONETOMANY
from .interfaces import PropComparator
from .interfaces import RelationshipDirection
from .interfaces import StrategizedProperty
from .util import _orm_annotate
from .util import _orm_deannotate
from .util import CascadeOptions
from .. import exc as sa_exc
from .. import Exists
from .. import log
from .. import schema
from .. import sql
from .. import util
from ..inspection import inspect
from ..sql import coercions
from ..sql import expression
from ..sql import operators
from ..sql import roles
from ..sql import visitors
from ..sql._typing import _ColumnExpressionArgument
from ..sql._typing import _HasClauseElement
from ..sql.annotation import _safe_annotate
from ..sql.elements import ColumnClause
from ..sql.elements import ColumnElement
from ..sql.util import _deep_annotate
from ..sql.util import _deep_deannotate
from ..sql.util import _shallow_annotate
from ..sql.util import adapt_criterion_to_null
from ..sql.util import ClauseAdapter
from ..sql.util import join_condition
from ..sql.util import selectables_overlap
from ..sql.util import visit_binary_product
from ..util.typing import de_optionalize_union_types
from ..util.typing import Literal
from ..util.typing import resolve_name_to_real_class_name
@log.class_logger
class RelationshipProperty(_IntrospectsAnnotations, StrategizedProperty[_T], log.Identified):
    """Describes an object property that holds a single item or list
    of items that correspond to a related database table.

    Public constructor is the :func:`_orm.relationship` function.

    .. seealso::

        :ref:`relationship_config_toplevel`

    """
    strategy_wildcard_key = strategy_options._RELATIONSHIP_TOKEN
    inherit_cache = True
    ':meta private:'
    _links_to_entity = True
    _is_relationship = True
    _overlaps: Sequence[str]
    _lazy_strategy: LazyLoader
    _persistence_only = dict(passive_deletes=False, passive_updates=True, enable_typechecks=True, active_history=False, cascade_backrefs=False)
    _dependency_processor: Optional[DependencyProcessor] = None
    primaryjoin: ColumnElement[bool]
    secondaryjoin: Optional[ColumnElement[bool]]
    secondary: Optional[FromClause]
    _join_condition: JoinCondition
    order_by: _RelationshipOrderByArg
    _user_defined_foreign_keys: Set[ColumnElement[Any]]
    _calculated_foreign_keys: Set[ColumnElement[Any]]
    remote_side: Set[ColumnElement[Any]]
    local_columns: Set[ColumnElement[Any]]
    synchronize_pairs: _ColumnPairs
    secondary_synchronize_pairs: Optional[_ColumnPairs]
    local_remote_pairs: Optional[_ColumnPairs]
    direction: RelationshipDirection
    _init_args: _RelationshipArgs

    def __init__(self, argument: Optional[_RelationshipArgumentType[_T]]=None, secondary: Optional[_RelationshipSecondaryArgument]=None, *, uselist: Optional[bool]=None, collection_class: Optional[Union[Type[Collection[Any]], Callable[[], Collection[Any]]]]=None, primaryjoin: Optional[_RelationshipJoinConditionArgument]=None, secondaryjoin: Optional[_RelationshipJoinConditionArgument]=None, back_populates: Optional[str]=None, order_by: _ORMOrderByArgument=False, backref: Optional[ORMBackrefArgument]=None, overlaps: Optional[str]=None, post_update: bool=False, cascade: str='save-update, merge', viewonly: bool=False, attribute_options: Optional[_AttributeOptions]=None, lazy: _LazyLoadArgumentType='select', passive_deletes: Union[Literal['all'], bool]=False, passive_updates: bool=True, active_history: bool=False, enable_typechecks: bool=True, foreign_keys: Optional[_ORMColCollectionArgument]=None, remote_side: Optional[_ORMColCollectionArgument]=None, join_depth: Optional[int]=None, comparator_factory: Optional[Type[RelationshipProperty.Comparator[Any]]]=None, single_parent: bool=False, innerjoin: bool=False, distinct_target_key: Optional[bool]=None, load_on_pending: bool=False, query_class: Optional[Type[Query[Any]]]=None, info: Optional[_InfoType]=None, omit_join: Literal[None, False]=None, sync_backref: Optional[bool]=None, doc: Optional[str]=None, bake_queries: Literal[True]=True, cascade_backrefs: Literal[False]=False, _local_remote_pairs: Optional[_ColumnPairs]=None, _legacy_inactive_history_style: bool=False):
        super().__init__(attribute_options=attribute_options)
        self.uselist = uselist
        self.argument = argument
        self._init_args = _RelationshipArgs(_RelationshipArg('secondary', secondary, None), _RelationshipArg('primaryjoin', primaryjoin, None), _RelationshipArg('secondaryjoin', secondaryjoin, None), _RelationshipArg('order_by', order_by, None), _RelationshipArg('foreign_keys', foreign_keys, None), _RelationshipArg('remote_side', remote_side, None))
        self.post_update = post_update
        self.viewonly = viewonly
        if viewonly:
            self._warn_for_persistence_only_flags(passive_deletes=passive_deletes, passive_updates=passive_updates, enable_typechecks=enable_typechecks, active_history=active_history, cascade_backrefs=cascade_backrefs)
        if viewonly and sync_backref:
            raise sa_exc.ArgumentError('sync_backref and viewonly cannot both be True')
        self.sync_backref = sync_backref
        self.lazy = lazy
        self.single_parent = single_parent
        self.collection_class = collection_class
        self.passive_deletes = passive_deletes
        if cascade_backrefs:
            raise sa_exc.ArgumentError("The 'cascade_backrefs' parameter passed to relationship() may only be set to False.")
        self.passive_updates = passive_updates
        self.enable_typechecks = enable_typechecks
        self.query_class = query_class
        self.innerjoin = innerjoin
        self.distinct_target_key = distinct_target_key
        self.doc = doc
        self.active_history = active_history
        self._legacy_inactive_history_style = _legacy_inactive_history_style
        self.join_depth = join_depth
        if omit_join:
            util.warn('setting omit_join to True is not supported; selectin loading of this relationship may not work correctly if this flag is set explicitly.  omit_join optimization is automatically detected for conditions under which it is supported.')
        self.omit_join = omit_join
        self.local_remote_pairs = _local_remote_pairs
        self.load_on_pending = load_on_pending
        self.comparator_factory = comparator_factory or RelationshipProperty.Comparator
        util.set_creation_order(self)
        if info is not None:
            self.info.update(info)
        self.strategy_key = (('lazy', self.lazy),)
        self._reverse_property: Set[RelationshipProperty[Any]] = set()
        if overlaps:
            self._overlaps = set(re.split('\\s*,\\s*', overlaps))
        else:
            self._overlaps = ()
        self.cascade = cascade
        self.back_populates = back_populates
        if self.back_populates:
            if backref:
                raise sa_exc.ArgumentError('backref and back_populates keyword arguments are mutually exclusive')
            self.backref = None
        else:
            self.backref = backref

    def _warn_for_persistence_only_flags(self, **kw: Any) -> None:
        for k, v in kw.items():
            if v != self._persistence_only[k]:
                util.warn('Setting %s on relationship() while also setting viewonly=True does not make sense, as a viewonly=True relationship does not perform persistence operations. This configuration may raise an error in a future release.' % (k,))

    def instrument_class(self, mapper: Mapper[Any]) -> None:
        attributes.register_descriptor(mapper.class_, self.key, comparator=self.comparator_factory(self, mapper), parententity=mapper, doc=self.doc)

    class Comparator(util.MemoizedSlots, PropComparator[_PT]):
        """Produce boolean, comparison, and other operators for
        :class:`.RelationshipProperty` attributes.

        See the documentation for :class:`.PropComparator` for a brief
        overview of ORM level operator definition.

        .. seealso::

            :class:`.PropComparator`

            :class:`.ColumnProperty.Comparator`

            :class:`.ColumnOperators`

            :ref:`types_operators`

            :attr:`.TypeEngine.comparator_factory`

        """
        __slots__ = ('entity', 'mapper', 'property', '_of_type', '_extra_criteria')
        prop: RODescriptorReference[RelationshipProperty[_PT]]
        _of_type: Optional[_EntityType[_PT]]

        def __init__(self, prop: RelationshipProperty[_PT], parentmapper: _InternalEntityType[Any], adapt_to_entity: Optional[AliasedInsp[Any]]=None, of_type: Optional[_EntityType[_PT]]=None, extra_criteria: Tuple[ColumnElement[bool], ...]=()):
            """Construction of :class:`.RelationshipProperty.Comparator`
            is internal to the ORM's attribute mechanics.

            """
            self.prop = prop
            self._parententity = parentmapper
            self._adapt_to_entity = adapt_to_entity
            if of_type:
                self._of_type = of_type
            else:
                self._of_type = None
            self._extra_criteria = extra_criteria

        def adapt_to_entity(self, adapt_to_entity: AliasedInsp[Any]) -> RelationshipProperty.Comparator[Any]:
            return self.__class__(self.prop, self._parententity, adapt_to_entity=adapt_to_entity, of_type=self._of_type)
        entity: _InternalEntityType[_PT]
        'The target entity referred to by this\n        :class:`.RelationshipProperty.Comparator`.\n\n        This is either a :class:`_orm.Mapper` or :class:`.AliasedInsp`\n        object.\n\n        This is the "target" or "remote" side of the\n        :func:`_orm.relationship`.\n\n        '
        mapper: Mapper[_PT]
        'The target :class:`_orm.Mapper` referred to by this\n        :class:`.RelationshipProperty.Comparator`.\n\n        This is the "target" or "remote" side of the\n        :func:`_orm.relationship`.\n\n        '

        def _memoized_attr_entity(self) -> _InternalEntityType[_PT]:
            if self._of_type:
                return inspect(self._of_type)
            else:
                return self.prop.entity

        def _memoized_attr_mapper(self) -> Mapper[_PT]:
            return self.entity.mapper

        def _source_selectable(self) -> FromClause:
            if self._adapt_to_entity:
                return self._adapt_to_entity.selectable
            else:
                return self.property.parent._with_polymorphic_selectable

        def __clause_element__(self) -> ColumnElement[bool]:
            adapt_from = self._source_selectable()
            if self._of_type:
                of_type_entity = inspect(self._of_type)
            else:
                of_type_entity = None
            pj, sj, source, dest, secondary, target_adapter = self.prop._create_joins(source_selectable=adapt_from, source_polymorphic=True, of_type_entity=of_type_entity, alias_secondary=True, extra_criteria=self._extra_criteria)
            if sj is not None:
                return pj & sj
            else:
                return pj

        def of_type(self, class_: _EntityType[Any]) -> PropComparator[_PT]:
            """Redefine this object in terms of a polymorphic subclass.

            See :meth:`.PropComparator.of_type` for an example.


            """
            return RelationshipProperty.Comparator(self.prop, self._parententity, adapt_to_entity=self._adapt_to_entity, of_type=class_, extra_criteria=self._extra_criteria)

        def and_(self, *criteria: _ColumnExpressionArgument[bool]) -> PropComparator[Any]:
            """Add AND criteria.

            See :meth:`.PropComparator.and_` for an example.

            .. versionadded:: 1.4

            """
            exprs = tuple((coercions.expect(roles.WhereHavingRole, clause) for clause in util.coerce_generator_arg(criteria)))
            return RelationshipProperty.Comparator(self.prop, self._parententity, adapt_to_entity=self._adapt_to_entity, of_type=self._of_type, extra_criteria=self._extra_criteria + exprs)

        def in_(self, other: Any) -> NoReturn:
            """Produce an IN clause - this is not implemented
            for :func:`_orm.relationship`-based attributes at this time.

            """
            raise NotImplementedError('in_() not yet supported for relationships.  For a simple many-to-one, use in_() against the set of foreign key values.')
        __hash__ = None

        def __eq__(self, other: Any) -> ColumnElement[bool]:
            """Implement the ``==`` operator.

            In a many-to-one context, such as::

              MyClass.some_prop == <some object>

            this will typically produce a
            clause such as::

              mytable.related_id == <some id>

            Where ``<some id>`` is the primary key of the given
            object.

            The ``==`` operator provides partial functionality for non-
            many-to-one comparisons:

            * Comparisons against collections are not supported.
              Use :meth:`~.Relationship.Comparator.contains`.
            * Compared to a scalar one-to-many, will produce a
              clause that compares the target columns in the parent to
              the given target.
            * Compared to a scalar many-to-many, an alias
              of the association table will be rendered as
              well, forming a natural join that is part of the
              main body of the query. This will not work for
              queries that go beyond simple AND conjunctions of
              comparisons, such as those which use OR. Use
              explicit joins, outerjoins, or
              :meth:`~.Relationship.Comparator.has` for
              more comprehensive non-many-to-one scalar
              membership tests.
            * Comparisons against ``None`` given in a one-to-many
              or many-to-many context produce a NOT EXISTS clause.

            """
            if other is None or isinstance(other, expression.Null):
                if self.property.direction in [ONETOMANY, MANYTOMANY]:
                    return ~self._criterion_exists()
                else:
                    return _orm_annotate(self.property._optimized_compare(None, adapt_source=self.adapter))
            elif self.property.uselist:
                raise sa_exc.InvalidRequestError("Can't compare a collection to an object or collection; use contains() to test for membership.")
            else:
                return _orm_annotate(self.property._optimized_compare(other, adapt_source=self.adapter))

        def _criterion_exists(self, criterion: Optional[_ColumnExpressionArgument[bool]]=None, **kwargs: Any) -> Exists:
            where_criteria = coercions.expect(roles.WhereHavingRole, criterion) if criterion is not None else None
            if getattr(self, '_of_type', None):
                info: Optional[_InternalEntityType[Any]] = inspect(self._of_type)
                assert info is not None
                target_mapper, to_selectable, is_aliased_class = (info.mapper, info.selectable, info.is_aliased_class)
                if self.property._is_self_referential and (not is_aliased_class):
                    to_selectable = to_selectable._anonymous_fromclause()
                single_crit = target_mapper._single_table_criterion
                if single_crit is not None:
                    if where_criteria is not None:
                        where_criteria = single_crit & where_criteria
                    else:
                        where_criteria = single_crit
            else:
                is_aliased_class = False
                to_selectable = None
            if self.adapter:
                source_selectable = self._source_selectable()
            else:
                source_selectable = None
            pj, sj, source, dest, secondary, target_adapter = self.property._create_joins(dest_selectable=to_selectable, source_selectable=source_selectable)
            for k in kwargs:
                crit = getattr(self.property.mapper.class_, k) == kwargs[k]
                if where_criteria is None:
                    where_criteria = crit
                else:
                    where_criteria = where_criteria & crit
            if sj is not None:
                j = _orm_annotate(pj) & sj
            else:
                j = _orm_annotate(pj, exclude=self.property.remote_side)
            if where_criteria is not None and target_adapter and (not is_aliased_class):
                where_criteria = target_adapter.traverse(where_criteria)
            if where_criteria is not None:
                where_criteria = where_criteria._annotate({'no_replacement_traverse': True})
            crit = j & sql.True_._ifnone(where_criteria)
            if secondary is not None:
                ex = sql.exists(1).where(crit).select_from(dest, secondary).correlate_except(dest, secondary)
            else:
                ex = sql.exists(1).where(crit).select_from(dest).correlate_except(dest)
            return ex

        def any(self, criterion: Optional[_ColumnExpressionArgument[bool]]=None, **kwargs: Any) -> ColumnElement[bool]:
            """Produce an expression that tests a collection against
            particular criterion, using EXISTS.

            An expression like::

                session.query(MyClass).filter(
                    MyClass.somereference.any(SomeRelated.x==2)
                )


            Will produce a query like::

                SELECT * FROM my_table WHERE
                EXISTS (SELECT 1 FROM related WHERE related.my_id=my_table.id
                AND related.x=2)

            Because :meth:`~.Relationship.Comparator.any` uses
            a correlated subquery, its performance is not nearly as
            good when compared against large target tables as that of
            using a join.

            :meth:`~.Relationship.Comparator.any` is particularly
            useful for testing for empty collections::

                session.query(MyClass).filter(
                    ~MyClass.somereference.any()
                )

            will produce::

                SELECT * FROM my_table WHERE
                NOT (EXISTS (SELECT 1 FROM related WHERE
                related.my_id=my_table.id))

            :meth:`~.Relationship.Comparator.any` is only
            valid for collections, i.e. a :func:`_orm.relationship`
            that has ``uselist=True``.  For scalar references,
            use :meth:`~.Relationship.Comparator.has`.

            """
            if not self.property.uselist:
                raise sa_exc.InvalidRequestError("'any()' not implemented for scalar attributes. Use has().")
            return self._criterion_exists(criterion, **kwargs)

        def has(self, criterion: Optional[_ColumnExpressionArgument[bool]]=None, **kwargs: Any) -> ColumnElement[bool]:
            """Produce an expression that tests a scalar reference against
            particular criterion, using EXISTS.

            An expression like::

                session.query(MyClass).filter(
                    MyClass.somereference.has(SomeRelated.x==2)
                )


            Will produce a query like::

                SELECT * FROM my_table WHERE
                EXISTS (SELECT 1 FROM related WHERE
                related.id==my_table.related_id AND related.x=2)

            Because :meth:`~.Relationship.Comparator.has` uses
            a correlated subquery, its performance is not nearly as
            good when compared against large target tables as that of
            using a join.

            :meth:`~.Relationship.Comparator.has` is only
            valid for scalar references, i.e. a :func:`_orm.relationship`
            that has ``uselist=False``.  For collection references,
            use :meth:`~.Relationship.Comparator.any`.

            """
            if self.property.uselist:
                raise sa_exc.InvalidRequestError("'has()' not implemented for collections. Use any().")
            return self._criterion_exists(criterion, **kwargs)

        def contains(self, other: _ColumnExpressionArgument[Any], **kwargs: Any) -> ColumnElement[bool]:
            """Return a simple expression that tests a collection for
            containment of a particular item.

            :meth:`~.Relationship.Comparator.contains` is
            only valid for a collection, i.e. a
            :func:`_orm.relationship` that implements
            one-to-many or many-to-many with ``uselist=True``.

            When used in a simple one-to-many context, an
            expression like::

                MyClass.contains(other)

            Produces a clause like::

                mytable.id == <some id>

            Where ``<some id>`` is the value of the foreign key
            attribute on ``other`` which refers to the primary
            key of its parent object. From this it follows that
            :meth:`~.Relationship.Comparator.contains` is
            very useful when used with simple one-to-many
            operations.

            For many-to-many operations, the behavior of
            :meth:`~.Relationship.Comparator.contains`
            has more caveats. The association table will be
            rendered in the statement, producing an "implicit"
            join, that is, includes multiple tables in the FROM
            clause which are equated in the WHERE clause::

                query(MyClass).filter(MyClass.contains(other))

            Produces a query like::

                SELECT * FROM my_table, my_association_table AS
                my_association_table_1 WHERE
                my_table.id = my_association_table_1.parent_id
                AND my_association_table_1.child_id = <some id>

            Where ``<some id>`` would be the primary key of
            ``other``. From the above, it is clear that
            :meth:`~.Relationship.Comparator.contains`
            will **not** work with many-to-many collections when
            used in queries that move beyond simple AND
            conjunctions, such as multiple
            :meth:`~.Relationship.Comparator.contains`
            expressions joined by OR. In such cases subqueries or
            explicit "outer joins" will need to be used instead.
            See :meth:`~.Relationship.Comparator.any` for
            a less-performant alternative using EXISTS, or refer
            to :meth:`_query.Query.outerjoin`
            as well as :ref:`orm_queryguide_joins`
            for more details on constructing outer joins.

            kwargs may be ignored by this operator but are required for API
            conformance.
            """
            if not self.prop.uselist:
                raise sa_exc.InvalidRequestError("'contains' not implemented for scalar attributes.  Use ==")
            clause = self.prop._optimized_compare(other, adapt_source=self.adapter)
            if self.prop.secondaryjoin is not None:
                clause.negation_clause = self.__negated_contains_or_equals(other)
            return clause

        def __negated_contains_or_equals(self, other: Any) -> ColumnElement[bool]:
            if self.prop.direction == MANYTOONE:
                state = attributes.instance_state(other)

                def state_bindparam(local_col: ColumnElement[Any], state: InstanceState[Any], remote_col: ColumnElement[Any]) -> BindParameter[Any]:
                    dict_ = state.dict
                    return sql.bindparam(local_col.key, type_=local_col.type, unique=True, callable_=self.prop._get_attr_w_warn_on_none(self.prop.mapper, state, dict_, remote_col))

                def adapt(col: _CE) -> _CE:
                    if self.adapter:
                        return self.adapter(col)
                    else:
                        return col
                if self.property._use_get:
                    return sql.and_(*[sql.or_(adapt(x) != state_bindparam(adapt(x), state, y), adapt(x) == None) for x, y in self.property.local_remote_pairs])
            criterion = sql.and_(*[x == y for x, y in zip(self.property.mapper.primary_key, self.property.mapper.primary_key_from_instance(other))])
            return ~self._criterion_exists(criterion)

        def __ne__(self, other: Any) -> ColumnElement[bool]:
            """Implement the ``!=`` operator.

            In a many-to-one context, such as::

              MyClass.some_prop != <some object>

            This will typically produce a clause such as::

              mytable.related_id != <some id>

            Where ``<some id>`` is the primary key of the
            given object.

            The ``!=`` operator provides partial functionality for non-
            many-to-one comparisons:

            * Comparisons against collections are not supported.
              Use
              :meth:`~.Relationship.Comparator.contains`
              in conjunction with :func:`_expression.not_`.
            * Compared to a scalar one-to-many, will produce a
              clause that compares the target columns in the parent to
              the given target.
            * Compared to a scalar many-to-many, an alias
              of the association table will be rendered as
              well, forming a natural join that is part of the
              main body of the query. This will not work for
              queries that go beyond simple AND conjunctions of
              comparisons, such as those which use OR. Use
              explicit joins, outerjoins, or
              :meth:`~.Relationship.Comparator.has` in
              conjunction with :func:`_expression.not_` for
              more comprehensive non-many-to-one scalar
              membership tests.
            * Comparisons against ``None`` given in a one-to-many
              or many-to-many context produce an EXISTS clause.

            """
            if other is None or isinstance(other, expression.Null):
                if self.property.direction == MANYTOONE:
                    return _orm_annotate(~self.property._optimized_compare(None, adapt_source=self.adapter))
                else:
                    return self._criterion_exists()
            elif self.property.uselist:
                raise sa_exc.InvalidRequestError("Can't compare a collection to an object or collection; use contains() to test for membership.")
            else:
                return _orm_annotate(self.__negated_contains_or_equals(other))

        def _memoized_attr_property(self) -> RelationshipProperty[_PT]:
            self.prop.parent._check_configure()
            return self.prop

    def _with_parent(self, instance: object, alias_secondary: bool=True, from_entity: Optional[_EntityType[Any]]=None) -> ColumnElement[bool]:
        assert instance is not None
        adapt_source: Optional[_CoreAdapterProto] = None
        if from_entity is not None:
            insp: Optional[_InternalEntityType[Any]] = inspect(from_entity)
            assert insp is not None
            if insp_is_aliased_class(insp):
                adapt_source = insp._adapter.adapt_clause
        return self._optimized_compare(instance, value_is_parent=True, adapt_source=adapt_source, alias_secondary=alias_secondary)

    def _optimized_compare(self, state: Any, value_is_parent: bool=False, adapt_source: Optional[_CoreAdapterProto]=None, alias_secondary: bool=True) -> ColumnElement[bool]:
        if state is not None:
            try:
                state = inspect(state)
            except sa_exc.NoInspectionAvailable:
                state = None
            if state is None or not getattr(state, 'is_instance', False):
                raise sa_exc.ArgumentError('Mapped instance expected for relationship comparison to object.   Classes, queries and other SQL elements are not accepted in this context; for comparison with a subquery, use %s.has(**criteria).' % self)
        reverse_direction = not value_is_parent
        if state is None:
            return self._lazy_none_clause(reverse_direction, adapt_source=adapt_source)
        if not reverse_direction:
            criterion, bind_to_col = (self._lazy_strategy._lazywhere, self._lazy_strategy._bind_to_col)
        else:
            criterion, bind_to_col = (self._lazy_strategy._rev_lazywhere, self._lazy_strategy._rev_bind_to_col)
        if reverse_direction:
            mapper = self.mapper
        else:
            mapper = self.parent
        dict_ = attributes.instance_dict(state.obj())

        def visit_bindparam(bindparam: BindParameter[Any]) -> None:
            if bindparam._identifying_key in bind_to_col:
                bindparam.callable = self._get_attr_w_warn_on_none(mapper, state, dict_, bind_to_col[bindparam._identifying_key])
        if self.secondary is not None and alias_secondary:
            criterion = ClauseAdapter(self.secondary._anonymous_fromclause()).traverse(criterion)
        criterion = visitors.cloned_traverse(criterion, {}, {'bindparam': visit_bindparam})
        if adapt_source:
            criterion = adapt_source(criterion)
        return criterion

    def _get_attr_w_warn_on_none(self, mapper: Mapper[Any], state: InstanceState[Any], dict_: _InstanceDict, column: ColumnElement[Any]) -> Callable[[], Any]:
        """Create the callable that is used in a many-to-one expression.

        E.g.::

            u1 = s.query(User).get(5)

            expr = Address.user == u1

        Above, the SQL should be "address.user_id = 5". The callable
        returned by this method produces the value "5" based on the identity
        of ``u1``.

        """
        prop = mapper.get_property_by_column(column)
        state._track_last_known_value(prop.key)
        lkv_fixed = state._last_known_values

        def _go() -> Any:
            assert lkv_fixed is not None
            last_known = to_return = lkv_fixed[prop.key]
            existing_is_available = last_known is not LoaderCallableStatus.NO_VALUE
            current_value = mapper._get_state_attr_by_column(state, dict_, column, passive=PassiveFlag.PASSIVE_OFF if state.persistent else PassiveFlag.PASSIVE_NO_FETCH ^ PassiveFlag.INIT_OK)
            if current_value is LoaderCallableStatus.NEVER_SET:
                if not existing_is_available:
                    raise sa_exc.InvalidRequestError("Can't resolve value for column %s on object %s; no value has been set for this column" % (column, state_str(state)))
            elif current_value is LoaderCallableStatus.PASSIVE_NO_RESULT:
                if not existing_is_available:
                    raise sa_exc.InvalidRequestError("Can't resolve value for column %s on object %s; the object is detached and the value was expired" % (column, state_str(state)))
            else:
                to_return = current_value
            if to_return is None:
                util.warn('Got None for value of column %s; this is unsupported for a relationship comparison and will not currently produce an IS comparison (but may in a future release)' % column)
            return to_return
        return _go

    def _lazy_none_clause(self, reverse_direction: bool=False, adapt_source: Optional[_CoreAdapterProto]=None) -> ColumnElement[bool]:
        if not reverse_direction:
            criterion, bind_to_col = (self._lazy_strategy._lazywhere, self._lazy_strategy._bind_to_col)
        else:
            criterion, bind_to_col = (self._lazy_strategy._rev_lazywhere, self._lazy_strategy._rev_bind_to_col)
        criterion = adapt_criterion_to_null(criterion, bind_to_col)
        if adapt_source:
            criterion = adapt_source(criterion)
        return criterion

    def __str__(self) -> str:
        return str(self.parent.class_.__name__) + '.' + self.key

    def merge(self, session: Session, source_state: InstanceState[Any], source_dict: _InstanceDict, dest_state: InstanceState[Any], dest_dict: _InstanceDict, load: bool, _recursive: Dict[Any, object], _resolve_conflict_map: Dict[_IdentityKeyType[Any], object]) -> None:
        if load:
            for r in self._reverse_property:
                if (source_state, r) in _recursive:
                    return
        if 'merge' not in self._cascade:
            return
        if self.key not in source_dict:
            return
        if self.uselist:
            impl = source_state.get_impl(self.key)
            assert is_has_collection_adapter(impl)
            instances_iterable = impl.get_collection(source_state, source_dict)
            assert not instances_iterable.empty if impl.collection else True
            if load:
                dest_state.get_impl(self.key).get(dest_state, dest_dict, passive=PassiveFlag.PASSIVE_MERGE)
            dest_list = []
            for current in instances_iterable:
                current_state = attributes.instance_state(current)
                current_dict = attributes.instance_dict(current)
                _recursive[current_state, self] = True
                obj = session._merge(current_state, current_dict, load=load, _recursive=_recursive, _resolve_conflict_map=_resolve_conflict_map)
                if obj is not None:
                    dest_list.append(obj)
            if not load:
                coll = attributes.init_state_collection(dest_state, dest_dict, self.key)
                for c in dest_list:
                    coll.append_without_event(c)
            else:
                dest_impl = dest_state.get_impl(self.key)
                assert is_has_collection_adapter(dest_impl)
                dest_impl.set(dest_state, dest_dict, dest_list, _adapt=False, passive=PassiveFlag.PASSIVE_MERGE)
        else:
            current = source_dict[self.key]
            if current is not None:
                current_state = attributes.instance_state(current)
                current_dict = attributes.instance_dict(current)
                _recursive[current_state, self] = True
                obj = session._merge(current_state, current_dict, load=load, _recursive=_recursive, _resolve_conflict_map=_resolve_conflict_map)
            else:
                obj = None
            if not load:
                dest_dict[self.key] = obj
            else:
                dest_state.get_impl(self.key).set(dest_state, dest_dict, obj, None)

    def _value_as_iterable(self, state: InstanceState[_O], dict_: _InstanceDict, key: str, passive: PassiveFlag=PassiveFlag.PASSIVE_OFF) -> Sequence[Tuple[InstanceState[_O], _O]]:
        """Return a list of tuples (state, obj) for the given
        key.

        returns an empty list if the value is None/empty/PASSIVE_NO_RESULT
        """
        impl = state.manager[key].impl
        x = impl.get(state, dict_, passive=passive)
        if x is LoaderCallableStatus.PASSIVE_NO_RESULT or x is None:
            return []
        elif is_has_collection_adapter(impl):
            return [(attributes.instance_state(o), o) for o in impl.get_collection(state, dict_, x, passive=passive)]
        else:
            return [(attributes.instance_state(x), x)]

    def cascade_iterator(self, type_: str, state: InstanceState[Any], dict_: _InstanceDict, visited_states: Set[InstanceState[Any]], halt_on: Optional[Callable[[InstanceState[Any]], bool]]=None) -> Iterator[Tuple[Any, Mapper[Any], InstanceState[Any], _InstanceDict]]:
        if type_ != 'delete' or self.passive_deletes:
            passive = PassiveFlag.PASSIVE_NO_INITIALIZE
        else:
            passive = PassiveFlag.PASSIVE_OFF | PassiveFlag.NO_RAISE
        if type_ == 'save-update':
            tuples = state.manager[self.key].impl.get_all_pending(state, dict_)
        else:
            tuples = self._value_as_iterable(state, dict_, self.key, passive=passive)
        skip_pending = type_ == 'refresh-expire' and 'delete-orphan' not in self._cascade
        for instance_state, c in tuples:
            if instance_state in visited_states:
                continue
            if c is None:
                continue
            assert instance_state is not None
            instance_dict = attributes.instance_dict(c)
            if halt_on and halt_on(instance_state):
                continue
            if skip_pending and (not instance_state.key):
                continue
            instance_mapper = instance_state.manager.mapper
            if not instance_mapper.isa(self.mapper.class_manager.mapper):
                raise AssertionError("Attribute '%s' on class '%s' doesn't handle objects of type '%s'" % (self.key, self.parent.class_, c.__class__))
            visited_states.add(instance_state)
            yield (c, instance_mapper, instance_state, instance_dict)

    @property
    def _effective_sync_backref(self) -> bool:
        if self.viewonly:
            return False
        else:
            return self.sync_backref is not False

    @staticmethod
    def _check_sync_backref(rel_a: RelationshipProperty[Any], rel_b: RelationshipProperty[Any]) -> None:
        if rel_a.viewonly and rel_b.sync_backref:
            raise sa_exc.InvalidRequestError('Relationship %s cannot specify sync_backref=True since %s includes viewonly=True.' % (rel_b, rel_a))
        if rel_a.viewonly and (not rel_b.viewonly) and (rel_b.sync_backref is not False):
            rel_b.sync_backref = False

    def _add_reverse_property(self, key: str) -> None:
        other = self.mapper.get_property(key, _configure_mappers=False)
        if not isinstance(other, RelationshipProperty):
            raise sa_exc.InvalidRequestError("back_populates on relationship '%s' refers to attribute '%s' that is not a relationship.  The back_populates parameter should refer to the name of a relationship on the target class." % (self, other))
        self._check_sync_backref(self, other)
        self._check_sync_backref(other, self)
        self._reverse_property.add(other)
        other._reverse_property.add(self)
        other._setup_entity()
        if not other.mapper.common_parent(self.parent):
            raise sa_exc.ArgumentError('reverse_property %r on relationship %s references relationship %s, which does not reference mapper %s' % (key, self, other, self.parent))
        if other._configure_started and self.direction in (ONETOMANY, MANYTOONE) and (self.direction == other.direction):
            raise sa_exc.ArgumentError('%s and back-reference %s are both of the same direction %r.  Did you mean to set remote_side on the many-to-one side ?' % (other, self, self.direction))

    @util.memoized_property
    def entity(self) -> _InternalEntityType[_T]:
        """Return the target mapped entity, which is an inspect() of the
        class or aliased class that is referenced by this
        :class:`.RelationshipProperty`.

        """
        self.parent._check_configure()
        return self.entity

    @util.memoized_property
    def mapper(self) -> Mapper[_T]:
        """Return the targeted :class:`_orm.Mapper` for this
        :class:`.RelationshipProperty`.

        """
        return self.entity.mapper

    def do_init(self) -> None:
        self._check_conflicts()
        self._process_dependent_arguments()
        self._setup_entity()
        self._setup_registry_dependencies()
        self._setup_join_conditions()
        self._check_cascade_settings(self._cascade)
        self._post_init()
        self._generate_backref()
        self._join_condition._warn_for_conflicting_sync_targets()
        super().do_init()
        self._lazy_strategy = cast('LazyLoader', self._get_strategy((('lazy', 'select'),)))

    def _setup_registry_dependencies(self) -> None:
        self.parent.mapper.registry._set_depends_on(self.entity.mapper.registry)

    def _process_dependent_arguments(self) -> None:
        """Convert incoming configuration arguments to their
        proper form.

        Callables are resolved, ORM annotations removed.

        """
        init_args = self._init_args
        for attr in ('order_by', 'primaryjoin', 'secondaryjoin', 'secondary', 'foreign_keys', 'remote_side'):
            rel_arg = getattr(init_args, attr)
            rel_arg._resolve_against_registry(self._clsregistry_resolvers[1])
        for attr in ('primaryjoin', 'secondaryjoin'):
            rel_arg = getattr(init_args, attr)
            val = rel_arg.resolved
            if val is not None:
                rel_arg.resolved = _orm_deannotate(coercions.expect(roles.ColumnArgumentRole, val, argname=attr))
        secondary = init_args.secondary.resolved
        if secondary is not None and _is_mapped_class(secondary):
            raise sa_exc.ArgumentError("secondary argument %s passed to to relationship() %s must be a Table object or other FROM clause; can't send a mapped class directly as rows in 'secondary' are persisted independently of a class that is mapped to that same table." % (secondary, self))
        if init_args.order_by.resolved is not False and init_args.order_by.resolved is not None:
            self.order_by = tuple((coercions.expect(roles.ColumnArgumentRole, x, argname='order_by') for x in util.to_list(init_args.order_by.resolved)))
        else:
            self.order_by = False
        self._user_defined_foreign_keys = util.column_set((coercions.expect(roles.ColumnArgumentRole, x, argname='foreign_keys') for x in util.to_column_set(init_args.foreign_keys.resolved)))
        self.remote_side = util.column_set((coercions.expect(roles.ColumnArgumentRole, x, argname='remote_side') for x in util.to_column_set(init_args.remote_side.resolved)))

    def declarative_scan(self, decl_scan: _ClassScanMapperConfig, registry: _RegistryType, cls: Type[Any], originating_module: Optional[str], key: str, mapped_container: Optional[Type[Mapped[Any]]], annotation: Optional[_AnnotationScanType], extracted_mapped_annotation: Optional[_AnnotationScanType], is_dataclass_field: bool) -> None:
        argument = extracted_mapped_annotation
        if extracted_mapped_annotation is None:
            if self.argument is None:
                self._raise_for_required(key, cls)
            else:
                return
        argument = extracted_mapped_annotation
        assert originating_module is not None
        if mapped_container is not None:
            is_write_only = issubclass(mapped_container, WriteOnlyMapped)
            is_dynamic = issubclass(mapped_container, DynamicMapped)
            if is_write_only:
                self.lazy = 'write_only'
                self.strategy_key = (('lazy', self.lazy),)
            elif is_dynamic:
                self.lazy = 'dynamic'
                self.strategy_key = (('lazy', self.lazy),)
        else:
            is_write_only = is_dynamic = False
        argument = de_optionalize_union_types(argument)
        if hasattr(argument, '__origin__'):
            arg_origin = argument.__origin__
            if isinstance(arg_origin, type) and issubclass(arg_origin, abc.Collection):
                if self.collection_class is None:
                    if _py_inspect.isabstract(arg_origin):
                        raise sa_exc.ArgumentError(f"Collection annotation type {arg_origin} cannot be instantiated; please provide an explicit 'collection_class' parameter (e.g. list, set, etc.) to the relationship() function to accompany this annotation")
                    self.collection_class = arg_origin
            elif not is_write_only and (not is_dynamic):
                self.uselist = False
            if argument.__args__:
                if isinstance(arg_origin, type) and issubclass(arg_origin, typing.Mapping):
                    type_arg = argument.__args__[-1]
                else:
                    type_arg = argument.__args__[0]
                if hasattr(type_arg, '__forward_arg__'):
                    str_argument = type_arg.__forward_arg__
                    argument = resolve_name_to_real_class_name(str_argument, originating_module)
                else:
                    argument = type_arg
            else:
                raise sa_exc.ArgumentError(f'Generic alias {argument} requires an argument')
        elif hasattr(argument, '__forward_arg__'):
            argument = argument.__forward_arg__
            argument = resolve_name_to_real_class_name(argument, originating_module)
        if self.collection_class is None and (not is_write_only) and (not is_dynamic):
            self.uselist = False
        if self.argument is None:
            self.argument = cast('_RelationshipArgumentType[_T]', argument)

    @util.preload_module('sqlalchemy.orm.mapper')
    def _setup_entity(self, __argument: Any=None) -> None:
        if 'entity' in self.__dict__:
            return
        mapperlib = util.preloaded.orm_mapper
        if __argument:
            argument = __argument
        else:
            argument = self.argument
        resolved_argument: _ExternalEntityType[Any]
        if isinstance(argument, str):
            resolved_argument = cast('_ExternalEntityType[Any]', self._clsregistry_resolve_name(argument)())
        elif callable(argument) and (not isinstance(argument, (type, mapperlib.Mapper))):
            resolved_argument = argument()
        else:
            resolved_argument = argument
        entity: _InternalEntityType[Any]
        if isinstance(resolved_argument, type):
            entity = class_mapper(resolved_argument, configure=False)
        else:
            try:
                entity = inspect(resolved_argument)
            except sa_exc.NoInspectionAvailable:
                entity = None
            if not hasattr(entity, 'mapper'):
                raise sa_exc.ArgumentError("relationship '%s' expects a class or a mapper argument (received: %s)" % (self.key, type(resolved_argument)))
        self.entity = entity
        self.target = self.entity.persist_selectable

    def _setup_join_conditions(self) -> None:
        self._join_condition = jc = JoinCondition(parent_persist_selectable=self.parent.persist_selectable, child_persist_selectable=self.entity.persist_selectable, parent_local_selectable=self.parent.local_table, child_local_selectable=self.entity.local_table, primaryjoin=self._init_args.primaryjoin.resolved, secondary=self._init_args.secondary.resolved, secondaryjoin=self._init_args.secondaryjoin.resolved, parent_equivalents=self.parent._equivalent_columns, child_equivalents=self.mapper._equivalent_columns, consider_as_foreign_keys=self._user_defined_foreign_keys, local_remote_pairs=self.local_remote_pairs, remote_side=self.remote_side, self_referential=self._is_self_referential, prop=self, support_sync=not self.viewonly, can_be_synced_fn=self._columns_are_mapped)
        self.primaryjoin = jc.primaryjoin
        self.secondaryjoin = jc.secondaryjoin
        self.secondary = jc.secondary
        self.direction = jc.direction
        self.local_remote_pairs = jc.local_remote_pairs
        self.remote_side = jc.remote_columns
        self.local_columns = jc.local_columns
        self.synchronize_pairs = jc.synchronize_pairs
        self._calculated_foreign_keys = jc.foreign_key_columns
        self.secondary_synchronize_pairs = jc.secondary_synchronize_pairs

    @property
    def _clsregistry_resolve_arg(self) -> Callable[[str, bool], _class_resolver]:
        return self._clsregistry_resolvers[1]

    @property
    def _clsregistry_resolve_name(self) -> Callable[[str], Callable[[], Union[Type[Any], Table, _ModNS]]]:
        return self._clsregistry_resolvers[0]

    @util.memoized_property
    @util.preload_module('sqlalchemy.orm.clsregistry')
    def _clsregistry_resolvers(self) -> Tuple[Callable[[str], Callable[[], Union[Type[Any], Table, _ModNS]]], Callable[[str, bool], _class_resolver]]:
        _resolver = util.preloaded.orm_clsregistry._resolver
        return _resolver(self.parent.class_, self)

    def _check_conflicts(self) -> None:
        """Test that this relationship is legal, warn about
        inheritance conflicts."""
        if self.parent.non_primary and (not class_mapper(self.parent.class_, configure=False).has_property(self.key)):
            raise sa_exc.ArgumentError("Attempting to assign a new relationship '%s' to a non-primary mapper on class '%s'.  New relationships can only be added to the primary mapper, i.e. the very first mapper created for class '%s' " % (self.key, self.parent.class_.__name__, self.parent.class_.__name__))

    @property
    def cascade(self) -> CascadeOptions:
        """Return the current cascade setting for this
        :class:`.RelationshipProperty`.
        """
        return self._cascade

    @cascade.setter
    def cascade(self, cascade: Union[str, CascadeOptions]) -> None:
        self._set_cascade(cascade)

    def _set_cascade(self, cascade_arg: Union[str, CascadeOptions]) -> None:
        cascade = CascadeOptions(cascade_arg)
        if self.viewonly:
            cascade = CascadeOptions(cascade.intersection(CascadeOptions._viewonly_cascades))
        if 'mapper' in self.__dict__:
            self._check_cascade_settings(cascade)
        self._cascade = cascade
        if self._dependency_processor:
            self._dependency_processor.cascade = cascade

    def _check_cascade_settings(self, cascade: CascadeOptions) -> None:
        if cascade.delete_orphan and (not self.single_parent) and (self.direction is MANYTOMANY or self.direction is MANYTOONE):
            raise sa_exc.ArgumentError('For %(direction)s relationship %(rel)s, delete-orphan cascade is normally configured only on the "one" side of a one-to-many relationship, and not on the "many" side of a many-to-one or many-to-many relationship.  To force this relationship to allow a particular "%(relatedcls)s" object to be referenced by only a single "%(clsname)s" object at a time via the %(rel)s relationship, which would allow delete-orphan cascade to take place in this direction, set the single_parent=True flag.' % {'rel': self, 'direction': 'many-to-one' if self.direction is MANYTOONE else 'many-to-many', 'clsname': self.parent.class_.__name__, 'relatedcls': self.mapper.class_.__name__}, code='bbf0')
        if self.passive_deletes == 'all' and ('delete' in cascade or 'delete-orphan' in cascade):
            raise sa_exc.ArgumentError("On %s, can't set passive_deletes='all' in conjunction with 'delete' or 'delete-orphan' cascade" % self)
        if cascade.delete_orphan:
            self.mapper.primary_mapper()._delete_orphans.append((self.key, self.parent.class_))

    def _persists_for(self, mapper: Mapper[Any]) -> bool:
        """Return True if this property will persist values on behalf
        of the given mapper.

        """
        return self.key in mapper.relationships and mapper.relationships[self.key] is self

    def _columns_are_mapped(self, *cols: ColumnElement[Any]) -> bool:
        """Return True if all columns in the given collection are
        mapped by the tables referenced by this :class:`.RelationshipProperty`.

        """
        secondary = self._init_args.secondary.resolved
        for c in cols:
            if secondary is not None and secondary.c.contains_column(c):
                continue
            if not self.parent.persist_selectable.c.contains_column(c) and (not self.target.c.contains_column(c)):
                return False
        return True

    def _generate_backref(self) -> None:
        """Interpret the 'backref' instruction to create a
        :func:`_orm.relationship` complementary to this one."""
        if self.parent.non_primary:
            return
        if self.backref is not None and (not self.back_populates):
            kwargs: Dict[str, Any]
            if isinstance(self.backref, str):
                backref_key, kwargs = (self.backref, {})
            else:
                backref_key, kwargs = self.backref
            mapper = self.mapper.primary_mapper()
            if not mapper.concrete:
                check = set(mapper.iterate_to_root()).union(mapper.self_and_descendants)
                for m in check:
                    if m.has_property(backref_key) and (not m.concrete):
                        raise sa_exc.ArgumentError("Error creating backref '%s' on relationship '%s': property of that name exists on mapper '%s'" % (backref_key, self, m))
            if self.secondary is not None:
                pj = kwargs.pop('primaryjoin', self._join_condition.secondaryjoin_minus_local)
                sj = kwargs.pop('secondaryjoin', self._join_condition.primaryjoin_minus_local)
            else:
                pj = kwargs.pop('primaryjoin', self._join_condition.primaryjoin_reverse_remote)
                sj = kwargs.pop('secondaryjoin', None)
                if sj:
                    raise sa_exc.InvalidRequestError("Can't assign 'secondaryjoin' on a backref against a non-secondary relationship.")
            foreign_keys = kwargs.pop('foreign_keys', self._user_defined_foreign_keys)
            parent = self.parent.primary_mapper()
            kwargs.setdefault('viewonly', self.viewonly)
            kwargs.setdefault('post_update', self.post_update)
            kwargs.setdefault('passive_updates', self.passive_updates)
            kwargs.setdefault('sync_backref', self.sync_backref)
            self.back_populates = backref_key
            relationship = RelationshipProperty(parent, self.secondary, primaryjoin=pj, secondaryjoin=sj, foreign_keys=foreign_keys, back_populates=self.key, **kwargs)
            mapper._configure_property(backref_key, relationship, warn_for_existing=True)
        if self.back_populates:
            self._add_reverse_property(self.back_populates)

    @util.preload_module('sqlalchemy.orm.dependency')
    def _post_init(self) -> None:
        dependency = util.preloaded.orm_dependency
        if self.uselist is None:
            self.uselist = self.direction is not MANYTOONE
        if not self.viewonly:
            self._dependency_processor = dependency.DependencyProcessor.from_relationship(self)

    @util.memoized_property
    def _use_get(self) -> bool:
        """memoize the 'use_get' attribute of this RelationshipLoader's
        lazyloader."""
        strategy = self._lazy_strategy
        return strategy.use_get

    @util.memoized_property
    def _is_self_referential(self) -> bool:
        return self.mapper.common_parent(self.parent)

    def _create_joins(self, source_polymorphic: bool=False, source_selectable: Optional[FromClause]=None, dest_selectable: Optional[FromClause]=None, of_type_entity: Optional[_InternalEntityType[Any]]=None, alias_secondary: bool=False, extra_criteria: Tuple[ColumnElement[bool], ...]=()) -> Tuple[ColumnElement[bool], Optional[ColumnElement[bool]], FromClause, FromClause, Optional[FromClause], Optional[ClauseAdapter]]:
        aliased = False
        if alias_secondary and self.secondary is not None:
            aliased = True
        if source_selectable is None:
            if source_polymorphic and self.parent.with_polymorphic:
                source_selectable = self.parent._with_polymorphic_selectable
        if of_type_entity:
            dest_mapper = of_type_entity.mapper
            if dest_selectable is None:
                dest_selectable = of_type_entity.selectable
                aliased = True
        else:
            dest_mapper = self.mapper
        if dest_selectable is None:
            dest_selectable = self.entity.selectable
            if self.mapper.with_polymorphic:
                aliased = True
            if self._is_self_referential and source_selectable is None:
                dest_selectable = dest_selectable._anonymous_fromclause()
                aliased = True
        elif dest_selectable is not self.mapper._with_polymorphic_selectable or self.mapper.with_polymorphic:
            aliased = True
        single_crit = dest_mapper._single_table_criterion
        aliased = aliased or (source_selectable is not None and (source_selectable is not self.parent._with_polymorphic_selectable or source_selectable._is_subquery))
        primaryjoin, secondaryjoin, secondary, target_adapter, dest_selectable = self._join_condition.join_targets(source_selectable, dest_selectable, aliased, single_crit, extra_criteria)
        if source_selectable is None:
            source_selectable = self.parent.local_table
        if dest_selectable is None:
            dest_selectable = self.entity.local_table
        return (primaryjoin, secondaryjoin, source_selectable, dest_selectable, secondary, target_adapter)