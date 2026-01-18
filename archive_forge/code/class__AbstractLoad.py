from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from . import util as orm_util
from ._typing import insp_is_aliased_class
from ._typing import insp_is_attribute
from ._typing import insp_is_mapper
from ._typing import insp_is_mapper_property
from .attributes import QueryableAttribute
from .base import InspectionAttr
from .interfaces import LoaderOption
from .path_registry import _DEFAULT_TOKEN
from .path_registry import _StrPathToken
from .path_registry import _WILDCARD_TOKEN
from .path_registry import AbstractEntityRegistry
from .path_registry import path_is_property
from .path_registry import PathRegistry
from .path_registry import TokenRegistry
from .util import _orm_full_deannotate
from .util import AliasedInsp
from .. import exc as sa_exc
from .. import inspect
from .. import util
from ..sql import and_
from ..sql import cache_key
from ..sql import coercions
from ..sql import roles
from ..sql import traversals
from ..sql import visitors
from ..sql.base import _generative
from ..util.typing import Final
from ..util.typing import Literal
from ..util.typing import Self
class _AbstractLoad(traversals.GenerativeOnTraversal, LoaderOption):
    __slots__ = ('propagate_to_loaders',)
    _is_strategy_option = True
    propagate_to_loaders: bool

    def contains_eager(self, attr: _AttrType, alias: Optional[_FromClauseArgument]=None, _is_chain: bool=False) -> Self:
        """Indicate that the given attribute should be eagerly loaded from
        columns stated manually in the query.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        The option is used in conjunction with an explicit join that loads
        the desired rows, i.e.::

            sess.query(Order).join(Order.user).options(
                contains_eager(Order.user)
            )

        The above query would join from the ``Order`` entity to its related
        ``User`` entity, and the returned ``Order`` objects would have the
        ``Order.user`` attribute pre-populated.

        It may also be used for customizing the entries in an eagerly loaded
        collection; queries will normally want to use the
        :ref:`orm_queryguide_populate_existing` execution option assuming the
        primary collection of parent objects may already have been loaded::

            sess.query(User).join(User.addresses).filter(
                Address.email_address.like("%@aol.com")
            ).options(contains_eager(User.addresses)).populate_existing()

        See the section :ref:`contains_eager` for complete usage details.

        .. seealso::

            :ref:`loading_toplevel`

            :ref:`contains_eager`

        """
        if alias is not None:
            if not isinstance(alias, str):
                coerced_alias = coercions.expect(roles.FromClauseRole, alias)
            else:
                util.warn_deprecated("Passing a string name for the 'alias' argument to 'contains_eager()` is deprecated, and will not work in a future release.  Please use a sqlalchemy.alias() or sqlalchemy.orm.aliased() construct.", version='1.4')
                coerced_alias = alias
        elif getattr(attr, '_of_type', None):
            assert isinstance(attr, QueryableAttribute)
            ot: Optional[_InternalEntityType[Any]] = inspect(attr._of_type)
            assert ot is not None
            coerced_alias = ot.selectable
        else:
            coerced_alias = None
        cloned = self._set_relationship_strategy(attr, {'lazy': 'joined'}, propagate_to_loaders=False, opts={'eager_from_alias': coerced_alias}, _reconcile_to_other=True if _is_chain else None)
        return cloned

    def load_only(self, *attrs: _AttrType, raiseload: bool=False) -> Self:
        """Indicate that for a particular entity, only the given list
        of column-based attribute names should be loaded; all others will be
        deferred.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        Example - given a class ``User``, load only the ``name`` and
        ``fullname`` attributes::

            session.query(User).options(load_only(User.name, User.fullname))

        Example - given a relationship ``User.addresses -> Address``, specify
        subquery loading for the ``User.addresses`` collection, but on each
        ``Address`` object load only the ``email_address`` attribute::

            session.query(User).options(
                subqueryload(User.addresses).load_only(Address.email_address)
            )

        For a statement that has multiple entities,
        the lead entity can be
        specifically referred to using the :class:`_orm.Load` constructor::

            stmt = (
                select(User, Address)
                .join(User.addresses)
                .options(
                    Load(User).load_only(User.name, User.fullname),
                    Load(Address).load_only(Address.email_address),
                )
            )

        When used together with the
        :ref:`populate_existing <orm_queryguide_populate_existing>`
        execution option only the attributes listed will be refreshed.

        :param \\*attrs: Attributes to be loaded, all others will be deferred.

        :param raiseload: raise :class:`.InvalidRequestError` rather than
         lazy loading a value when a deferred attribute is accessed. Used
         to prevent unwanted SQL from being emitted.

         .. versionadded:: 2.0

        .. seealso::

            :ref:`orm_queryguide_column_deferral` - in the
            :ref:`queryguide_toplevel`

        :param \\*attrs: Attributes to be loaded, all others will be deferred.

        :param raiseload: raise :class:`.InvalidRequestError` rather than
         lazy loading a value when a deferred attribute is accessed. Used
         to prevent unwanted SQL from being emitted.

         .. versionadded:: 2.0

        """
        cloned = self._set_column_strategy(attrs, {'deferred': False, 'instrument': True})
        wildcard_strategy = {'deferred': True, 'instrument': True}
        if raiseload:
            wildcard_strategy['raiseload'] = True
        cloned = cloned._set_column_strategy(('*',), wildcard_strategy)
        return cloned

    def joinedload(self, attr: _AttrType, innerjoin: Optional[bool]=None) -> Self:
        """Indicate that the given attribute should be loaded using joined
        eager loading.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        examples::

            # joined-load the "orders" collection on "User"
            select(User).options(joinedload(User.orders))

            # joined-load Order.items and then Item.keywords
            select(Order).options(
                joinedload(Order.items).joinedload(Item.keywords)
            )

            # lazily load Order.items, but when Items are loaded,
            # joined-load the keywords collection
            select(Order).options(
                lazyload(Order.items).joinedload(Item.keywords)
            )

        :param innerjoin: if ``True``, indicates that the joined eager load
         should use an inner join instead of the default of left outer join::

            select(Order).options(joinedload(Order.user, innerjoin=True))

        In order to chain multiple eager joins together where some may be
        OUTER and others INNER, right-nested joins are used to link them::

            select(A).options(
                joinedload(A.bs, innerjoin=False).joinedload(
                    B.cs, innerjoin=True
                )
            )

        The above query, linking A.bs via "outer" join and B.cs via "inner"
        join would render the joins as "a LEFT OUTER JOIN (b JOIN c)". When
        using older versions of SQLite (< 3.7.16), this form of JOIN is
        translated to use full subqueries as this syntax is otherwise not
        directly supported.

        The ``innerjoin`` flag can also be stated with the term ``"unnested"``.
        This indicates that an INNER JOIN should be used, *unless* the join
        is linked to a LEFT OUTER JOIN to the left, in which case it
        will render as LEFT OUTER JOIN.  For example, supposing ``A.bs``
        is an outerjoin::

            select(A).options(
                joinedload(A.bs).joinedload(B.cs, innerjoin="unnested")
            )


        The above join will render as "a LEFT OUTER JOIN b LEFT OUTER JOIN c",
        rather than as "a LEFT OUTER JOIN (b JOIN c)".

        .. note:: The "unnested" flag does **not** affect the JOIN rendered
            from a many-to-many association table, e.g. a table configured as
            :paramref:`_orm.relationship.secondary`, to the target table; for
            correctness of results, these joins are always INNER and are
            therefore right-nested if linked to an OUTER join.

        .. note::

            The joins produced by :func:`_orm.joinedload` are **anonymously
            aliased**. The criteria by which the join proceeds cannot be
            modified, nor can the ORM-enabled :class:`_sql.Select` or legacy
            :class:`_query.Query` refer to these joins in any way, including
            ordering. See :ref:`zen_of_eager_loading` for further detail.

            To produce a specific SQL JOIN which is explicitly available, use
            :meth:`_sql.Select.join` and :meth:`_query.Query.join`. To combine
            explicit JOINs with eager loading of collections, use
            :func:`_orm.contains_eager`; see :ref:`contains_eager`.

        .. seealso::

            :ref:`loading_toplevel`

            :ref:`joined_eager_loading`

        """
        loader = self._set_relationship_strategy(attr, {'lazy': 'joined'}, opts={'innerjoin': innerjoin} if innerjoin is not None else util.EMPTY_DICT)
        return loader

    def subqueryload(self, attr: _AttrType) -> Self:
        """Indicate that the given attribute should be loaded using
        subquery eager loading.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        examples::

            # subquery-load the "orders" collection on "User"
            select(User).options(subqueryload(User.orders))

            # subquery-load Order.items and then Item.keywords
            select(Order).options(
                subqueryload(Order.items).subqueryload(Item.keywords)
            )

            # lazily load Order.items, but when Items are loaded,
            # subquery-load the keywords collection
            select(Order).options(
                lazyload(Order.items).subqueryload(Item.keywords)
            )


        .. seealso::

            :ref:`loading_toplevel`

            :ref:`subquery_eager_loading`

        """
        return self._set_relationship_strategy(attr, {'lazy': 'subquery'})

    def selectinload(self, attr: _AttrType, recursion_depth: Optional[int]=None) -> Self:
        """Indicate that the given attribute should be loaded using
        SELECT IN eager loading.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        examples::

            # selectin-load the "orders" collection on "User"
            select(User).options(selectinload(User.orders))

            # selectin-load Order.items and then Item.keywords
            select(Order).options(
                selectinload(Order.items).selectinload(Item.keywords)
            )

            # lazily load Order.items, but when Items are loaded,
            # selectin-load the keywords collection
            select(Order).options(
                lazyload(Order.items).selectinload(Item.keywords)
            )

        :param recursion_depth: optional int; when set to a positive integer
         in conjunction with a self-referential relationship,
         indicates "selectin" loading will continue that many levels deep
         automatically until no items are found.

         .. note:: The :paramref:`_orm.selectinload.recursion_depth` option
            currently supports only self-referential relationships.  There
            is not yet an option to automatically traverse recursive structures
            with more than one relationship involved.

            Additionally, the :paramref:`_orm.selectinload.recursion_depth`
            parameter is new and experimental and should be treated as "alpha"
            status for the 2.0 series.

         .. versionadded:: 2.0 added
            :paramref:`_orm.selectinload.recursion_depth`


        .. seealso::

            :ref:`loading_toplevel`

            :ref:`selectin_eager_loading`

        """
        return self._set_relationship_strategy(attr, {'lazy': 'selectin'}, opts={'recursion_depth': recursion_depth})

    def lazyload(self, attr: _AttrType) -> Self:
        """Indicate that the given attribute should be loaded using "lazy"
        loading.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        .. seealso::

            :ref:`loading_toplevel`

            :ref:`lazy_loading`

        """
        return self._set_relationship_strategy(attr, {'lazy': 'select'})

    def immediateload(self, attr: _AttrType, recursion_depth: Optional[int]=None) -> Self:
        """Indicate that the given attribute should be loaded using
        an immediate load with a per-attribute SELECT statement.

        The load is achieved using the "lazyloader" strategy and does not
        fire off any additional eager loaders.

        The :func:`.immediateload` option is superseded in general
        by the :func:`.selectinload` option, which performs the same task
        more efficiently by emitting a SELECT for all loaded objects.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        :param recursion_depth: optional int; when set to a positive integer
         in conjunction with a self-referential relationship,
         indicates "selectin" loading will continue that many levels deep
         automatically until no items are found.

         .. note:: The :paramref:`_orm.immediateload.recursion_depth` option
            currently supports only self-referential relationships.  There
            is not yet an option to automatically traverse recursive structures
            with more than one relationship involved.

         .. warning:: This parameter is new and experimental and should be
            treated as "alpha" status

         .. versionadded:: 2.0 added
            :paramref:`_orm.immediateload.recursion_depth`


        .. seealso::

            :ref:`loading_toplevel`

            :ref:`selectin_eager_loading`

        """
        loader = self._set_relationship_strategy(attr, {'lazy': 'immediate'}, opts={'recursion_depth': recursion_depth})
        return loader

    def noload(self, attr: _AttrType) -> Self:
        """Indicate that the given relationship attribute should remain
        unloaded.

        The relationship attribute will return ``None`` when accessed without
        producing any loading effect.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        :func:`_orm.noload` applies to :func:`_orm.relationship` attributes
        only.

        .. note:: Setting this loading strategy as the default strategy
            for a relationship using the :paramref:`.orm.relationship.lazy`
            parameter may cause issues with flushes, such if a delete operation
            needs to load related objects and instead ``None`` was returned.

        .. seealso::

            :ref:`loading_toplevel`

        """
        return self._set_relationship_strategy(attr, {'lazy': 'noload'})

    def raiseload(self, attr: _AttrType, sql_only: bool=False) -> Self:
        """Indicate that the given attribute should raise an error if accessed.

        A relationship attribute configured with :func:`_orm.raiseload` will
        raise an :exc:`~sqlalchemy.exc.InvalidRequestError` upon access. The
        typical way this is useful is when an application is attempting to
        ensure that all relationship attributes that are accessed in a
        particular context would have been already loaded via eager loading.
        Instead of having to read through SQL logs to ensure lazy loads aren't
        occurring, this strategy will cause them to raise immediately.

        :func:`_orm.raiseload` applies to :func:`_orm.relationship` attributes
        only. In order to apply raise-on-SQL behavior to a column-based
        attribute, use the :paramref:`.orm.defer.raiseload` parameter on the
        :func:`.defer` loader option.

        :param sql_only: if True, raise only if the lazy load would emit SQL,
         but not if it is only checking the identity map, or determining that
         the related value should just be None due to missing keys. When False,
         the strategy will raise for all varieties of relationship loading.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        .. seealso::

            :ref:`loading_toplevel`

            :ref:`prevent_lazy_with_raiseload`

            :ref:`orm_queryguide_deferred_raiseload`

        """
        return self._set_relationship_strategy(attr, {'lazy': 'raise_on_sql' if sql_only else 'raise'})

    def defaultload(self, attr: _AttrType) -> Self:
        """Indicate an attribute should load using its predefined loader style.

        The behavior of this loading option is to not change the current
        loading style of the attribute, meaning that the previously configured
        one is used or, if no previous style was selected, the default
        loading will be used.

        This method is used to link to other loader options further into
        a chain of attributes without altering the loader style of the links
        along the chain.  For example, to set joined eager loading for an
        element of an element::

            session.query(MyClass).options(
                defaultload(MyClass.someattribute).joinedload(
                    MyOtherClass.someotherattribute
                )
            )

        :func:`.defaultload` is also useful for setting column-level options on
        a related class, namely that of :func:`.defer` and :func:`.undefer`::

            session.scalars(
                select(MyClass).options(
                    defaultload(MyClass.someattribute)
                    .defer("some_column")
                    .undefer("some_other_column")
                )
            )

        .. seealso::

            :ref:`orm_queryguide_relationship_sub_options`

            :meth:`_orm.Load.options`

        """
        return self._set_relationship_strategy(attr, None)

    def defer(self, key: _AttrType, raiseload: bool=False) -> Self:
        """Indicate that the given column-oriented attribute should be
        deferred, e.g. not loaded until accessed.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        e.g.::

            from sqlalchemy.orm import defer

            session.query(MyClass).options(
                defer(MyClass.attribute_one),
                defer(MyClass.attribute_two)
            )

        To specify a deferred load of an attribute on a related class,
        the path can be specified one token at a time, specifying the loading
        style for each link along the chain.  To leave the loading style
        for a link unchanged, use :func:`_orm.defaultload`::

            session.query(MyClass).options(
                defaultload(MyClass.someattr).defer(RelatedClass.some_column)
            )

        Multiple deferral options related to a relationship can be bundled
        at once using :meth:`_orm.Load.options`::


            select(MyClass).options(
                defaultload(MyClass.someattr).options(
                    defer(RelatedClass.some_column),
                    defer(RelatedClass.some_other_column),
                    defer(RelatedClass.another_column)
                )
            )

        :param key: Attribute to be deferred.

        :param raiseload: raise :class:`.InvalidRequestError` rather than
         lazy loading a value when the deferred attribute is accessed. Used
         to prevent unwanted SQL from being emitted.

        .. versionadded:: 1.4

        .. seealso::

            :ref:`orm_queryguide_column_deferral` - in the
            :ref:`queryguide_toplevel`

            :func:`_orm.load_only`

            :func:`_orm.undefer`

        """
        strategy = {'deferred': True, 'instrument': True}
        if raiseload:
            strategy['raiseload'] = True
        return self._set_column_strategy((key,), strategy)

    def undefer(self, key: _AttrType) -> Self:
        """Indicate that the given column-oriented attribute should be
        undeferred, e.g. specified within the SELECT statement of the entity
        as a whole.

        The column being undeferred is typically set up on the mapping as a
        :func:`.deferred` attribute.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        Examples::

            # undefer two columns
            session.query(MyClass).options(
                undefer(MyClass.col1), undefer(MyClass.col2)
            )

            # undefer all columns specific to a single class using Load + *
            session.query(MyClass, MyOtherClass).options(
                Load(MyClass).undefer("*")
            )

            # undefer a column on a related object
            select(MyClass).options(
                defaultload(MyClass.items).undefer(MyClass.text)
            )

        :param key: Attribute to be undeferred.

        .. seealso::

            :ref:`orm_queryguide_column_deferral` - in the
            :ref:`queryguide_toplevel`

            :func:`_orm.defer`

            :func:`_orm.undefer_group`

        """
        return self._set_column_strategy((key,), {'deferred': False, 'instrument': True})

    def undefer_group(self, name: str) -> Self:
        """Indicate that columns within the given deferred group name should be
        undeferred.

        The columns being undeferred are set up on the mapping as
        :func:`.deferred` attributes and include a "group" name.

        E.g::

            session.query(MyClass).options(undefer_group("large_attrs"))

        To undefer a group of attributes on a related entity, the path can be
        spelled out using relationship loader options, such as
        :func:`_orm.defaultload`::

            select(MyClass).options(
                defaultload("someattr").undefer_group("large_attrs")
            )

        .. seealso::

            :ref:`orm_queryguide_column_deferral` - in the
            :ref:`queryguide_toplevel`

            :func:`_orm.defer`

            :func:`_orm.undefer`

        """
        return self._set_column_strategy((_WILDCARD_TOKEN,), None, {f'undefer_group_{name}': True})

    def with_expression(self, key: _AttrType, expression: _ColumnExpressionArgument[Any]) -> Self:
        """Apply an ad-hoc SQL expression to a "deferred expression"
        attribute.

        This option is used in conjunction with the
        :func:`_orm.query_expression` mapper-level construct that indicates an
        attribute which should be the target of an ad-hoc SQL expression.

        E.g.::

            stmt = select(SomeClass).options(
                with_expression(SomeClass.x_y_expr, SomeClass.x + SomeClass.y)
            )

        .. versionadded:: 1.2

        :param key: Attribute to be populated

        :param expr: SQL expression to be applied to the attribute.

        .. seealso::

            :ref:`orm_queryguide_with_expression` - background and usage
            examples

        """
        expression = _orm_full_deannotate(coercions.expect(roles.LabeledColumnExprRole, expression))
        return self._set_column_strategy((key,), {'query_expression': True}, extra_criteria=(expression,))

    def selectin_polymorphic(self, classes: Iterable[Type[Any]]) -> Self:
        """Indicate an eager load should take place for all attributes
        specific to a subclass.

        This uses an additional SELECT with IN against all matched primary
        key values, and is the per-query analogue to the ``"selectin"``
        setting on the :paramref:`.mapper.polymorphic_load` parameter.

        .. versionadded:: 1.2

        .. seealso::

            :ref:`polymorphic_selectin`

        """
        self = self._set_class_strategy({'selectinload_polymorphic': True}, opts={'entities': tuple(sorted((inspect(cls) for cls in classes), key=id))})
        return self

    @overload
    def _coerce_strat(self, strategy: _StrategySpec) -> _StrategyKey:
        ...

    @overload
    def _coerce_strat(self, strategy: Literal[None]) -> None:
        ...

    def _coerce_strat(self, strategy: Optional[_StrategySpec]) -> Optional[_StrategyKey]:
        if strategy is not None:
            strategy_key = tuple(sorted(strategy.items()))
        else:
            strategy_key = None
        return strategy_key

    @_generative
    def _set_relationship_strategy(self, attr: _AttrType, strategy: Optional[_StrategySpec], propagate_to_loaders: bool=True, opts: Optional[_OptsType]=None, _reconcile_to_other: Optional[bool]=None) -> Self:
        strategy_key = self._coerce_strat(strategy)
        self._clone_for_bind_strategy((attr,), strategy_key, _RELATIONSHIP_TOKEN, opts=opts, propagate_to_loaders=propagate_to_loaders, reconcile_to_other=_reconcile_to_other)
        return self

    @_generative
    def _set_column_strategy(self, attrs: Tuple[_AttrType, ...], strategy: Optional[_StrategySpec], opts: Optional[_OptsType]=None, extra_criteria: Optional[Tuple[Any, ...]]=None) -> Self:
        strategy_key = self._coerce_strat(strategy)
        self._clone_for_bind_strategy(attrs, strategy_key, _COLUMN_TOKEN, opts=opts, attr_group=attrs, extra_criteria=extra_criteria)
        return self

    @_generative
    def _set_generic_strategy(self, attrs: Tuple[_AttrType, ...], strategy: _StrategySpec, _reconcile_to_other: Optional[bool]=None) -> Self:
        strategy_key = self._coerce_strat(strategy)
        self._clone_for_bind_strategy(attrs, strategy_key, None, propagate_to_loaders=True, reconcile_to_other=_reconcile_to_other)
        return self

    @_generative
    def _set_class_strategy(self, strategy: _StrategySpec, opts: _OptsType) -> Self:
        strategy_key = self._coerce_strat(strategy)
        self._clone_for_bind_strategy(None, strategy_key, None, opts=opts)
        return self

    def _apply_to_parent(self, parent: Load) -> None:
        """apply this :class:`_orm._AbstractLoad` object as a sub-option o
        a :class:`_orm.Load` object.

        Implementation is provided by subclasses.

        """
        raise NotImplementedError()

    def options(self, *opts: _AbstractLoad) -> Self:
        """Apply a series of options as sub-options to this
        :class:`_orm._AbstractLoad` object.

        Implementation is provided by subclasses.

        """
        raise NotImplementedError()

    def _clone_for_bind_strategy(self, attrs: Optional[Tuple[_AttrType, ...]], strategy: Optional[_StrategyKey], wildcard_key: Optional[_WildcardKeyType], opts: Optional[_OptsType]=None, attr_group: Optional[_AttrGroupType]=None, propagate_to_loaders: bool=True, reconcile_to_other: Optional[bool]=None, extra_criteria: Optional[Tuple[Any, ...]]=None) -> Self:
        raise NotImplementedError()

    def process_compile_state_replaced_entities(self, compile_state: ORMCompileState, mapper_entities: Sequence[_MapperEntity]) -> None:
        if not compile_state.compile_options._enable_eagerloads:
            return
        self._process(compile_state, mapper_entities, not bool(compile_state.current_path))

    def process_compile_state(self, compile_state: ORMCompileState) -> None:
        if not compile_state.compile_options._enable_eagerloads:
            return
        self._process(compile_state, compile_state._lead_mapper_entities, not bool(compile_state.current_path) and (not compile_state.compile_options._for_refresh_state))

    def _process(self, compile_state: ORMCompileState, mapper_entities: Sequence[_MapperEntity], raiseerr: bool) -> None:
        """implemented by subclasses"""
        raise NotImplementedError()

    @classmethod
    def _chop_path(cls, to_chop: _PathRepresentation, path: PathRegistry, debug: bool=False) -> Optional[_PathRepresentation]:
        i = -1
        for i, (c_token, p_token) in enumerate(zip(to_chop, path.natural_path)):
            if isinstance(c_token, str):
                if i == 0 and (c_token.endswith(f':{_DEFAULT_TOKEN}') or c_token.endswith(f':{_WILDCARD_TOKEN}')):
                    return to_chop
                elif c_token != f'{_RELATIONSHIP_TOKEN}:{_WILDCARD_TOKEN}' and c_token != p_token.key:
                    return None
            if c_token is p_token:
                continue
            elif isinstance(c_token, InspectionAttr) and insp_is_mapper(c_token) and insp_is_mapper(p_token) and c_token.isa(p_token):
                continue
            else:
                return None
        return to_chop[i + 1:]