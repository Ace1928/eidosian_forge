from __future__ import annotations
import collections
import itertools
from typing import Any
from typing import Dict
from typing import Tuple
from typing import TYPE_CHECKING
from . import attributes
from . import exc as orm_exc
from . import interfaces
from . import loading
from . import path_registry
from . import properties
from . import query
from . import relationships
from . import unitofwork
from . import util as orm_util
from .base import _DEFER_FOR_STATE
from .base import _RAISE_FOR_STATE
from .base import _SET_DEFERRED_EXPIRED
from .base import ATTR_WAS_SET
from .base import LoaderCallableStatus
from .base import PASSIVE_OFF
from .base import PassiveFlag
from .context import _column_descriptions
from .context import ORMCompileState
from .context import ORMSelectCompileState
from .context import QueryContext
from .interfaces import LoaderStrategy
from .interfaces import StrategizedProperty
from .session import _state_session
from .state import InstanceState
from .strategy_options import Load
from .util import _none_set
from .util import AliasedClass
from .. import event
from .. import exc as sa_exc
from .. import inspect
from .. import log
from .. import sql
from .. import util
from ..sql import util as sql_util
from ..sql import visitors
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import Select
@log.class_logger
@relationships.RelationshipProperty.strategy_for(lazy='joined')
@relationships.RelationshipProperty.strategy_for(lazy=False)
class JoinedLoader(AbstractRelationshipLoader):
    """Provide loading behavior for a :class:`.Relationship`
    using joined eager loading.

    """
    __slots__ = 'join_depth'

    def __init__(self, parent, strategy_key):
        super().__init__(parent, strategy_key)
        self.join_depth = self.parent_property.join_depth

    def init_class_attribute(self, mapper):
        self.parent_property._get_strategy((('lazy', 'select'),)).init_class_attribute(mapper)

    def setup_query(self, compile_state, query_entity, path, loadopt, adapter, column_collection=None, parentmapper=None, chained_from_outerjoin=False, **kwargs):
        """Add a left outer join to the statement that's being constructed."""
        if not compile_state.compile_options._enable_eagerloads:
            return
        elif self.uselist:
            compile_state.multi_row_eager_loaders = True
        path = path[self.parent_property]
        with_polymorphic = None
        user_defined_adapter = self._init_user_defined_eager_proc(loadopt, compile_state, compile_state.attributes) if loadopt else False
        if user_defined_adapter is not False:
            clauses, adapter, add_to_collection = self._setup_query_on_user_defined_adapter(compile_state, query_entity, path, adapter, user_defined_adapter)
        else:
            if not path.contains(compile_state.attributes, 'loader'):
                if self.join_depth:
                    if path.length / 2 > self.join_depth:
                        return
                elif path.contains_mapper(self.mapper):
                    return
            clauses, adapter, add_to_collection, chained_from_outerjoin = self._generate_row_adapter(compile_state, query_entity, path, loadopt, adapter, column_collection, parentmapper, chained_from_outerjoin)
            compile_state.eager_adding_joins = True
        with_poly_entity = path.get(compile_state.attributes, 'path_with_polymorphic', None)
        if with_poly_entity is not None:
            with_polymorphic = inspect(with_poly_entity).with_polymorphic_mappers
        else:
            with_polymorphic = None
        path = path[self.entity]
        loading._setup_entity_query(compile_state, self.mapper, query_entity, path, clauses, add_to_collection, with_polymorphic=with_polymorphic, parentmapper=self.mapper, chained_from_outerjoin=chained_from_outerjoin)
        has_nones = util.NONE_SET.intersection(compile_state.secondary_columns)
        if has_nones:
            if with_poly_entity is not None:
                raise sa_exc.InvalidRequestError('Detected unaliased columns when generating joined load.  Make sure to use aliased=True or flat=True when using joined loading with with_polymorphic().')
            else:
                compile_state.secondary_columns = [c for c in compile_state.secondary_columns if c is not None]

    def _init_user_defined_eager_proc(self, loadopt, compile_state, target_attributes):
        if 'eager_from_alias' not in loadopt.local_opts:
            return False
        path = loadopt.path.parent
        adapter = path.get(compile_state.attributes, 'user_defined_eager_row_processor', False)
        if adapter is not False:
            return adapter
        alias = loadopt.local_opts['eager_from_alias']
        root_mapper, prop = path[-2:]
        if alias is not None:
            if isinstance(alias, str):
                alias = prop.target.alias(alias)
            adapter = orm_util.ORMAdapter(orm_util._TraceAdaptRole.JOINEDLOAD_USER_DEFINED_ALIAS, prop.mapper, selectable=alias, equivalents=prop.mapper._equivalent_columns, limit_on_entity=False)
        elif path.contains(compile_state.attributes, 'path_with_polymorphic'):
            with_poly_entity = path.get(compile_state.attributes, 'path_with_polymorphic')
            adapter = orm_util.ORMAdapter(orm_util._TraceAdaptRole.JOINEDLOAD_PATH_WITH_POLYMORPHIC, with_poly_entity, equivalents=prop.mapper._equivalent_columns)
        else:
            adapter = compile_state._polymorphic_adapters.get(prop.mapper, None)
        path.set(target_attributes, 'user_defined_eager_row_processor', adapter)
        return adapter

    def _setup_query_on_user_defined_adapter(self, context, entity, path, adapter, user_defined_adapter):
        adapter = entity._get_entity_clauses(context)
        if adapter and user_defined_adapter:
            user_defined_adapter = user_defined_adapter.wrap(adapter)
            path.set(context.attributes, 'user_defined_eager_row_processor', user_defined_adapter)
        elif adapter:
            user_defined_adapter = adapter
            path.set(context.attributes, 'user_defined_eager_row_processor', user_defined_adapter)
        add_to_collection = context.primary_columns
        return (user_defined_adapter, adapter, add_to_collection)

    def _generate_row_adapter(self, compile_state, entity, path, loadopt, adapter, column_collection, parentmapper, chained_from_outerjoin):
        with_poly_entity = path.get(compile_state.attributes, 'path_with_polymorphic', None)
        if with_poly_entity:
            to_adapt = with_poly_entity
        else:
            insp = inspect(self.entity)
            if insp.is_aliased_class:
                alt_selectable = insp.selectable
            else:
                alt_selectable = None
            to_adapt = orm_util.AliasedClass(self.mapper, alias=alt_selectable._anonymous_fromclause(flat=True) if alt_selectable is not None else None, flat=True, use_mapper_path=True)
        to_adapt_insp = inspect(to_adapt)
        clauses = to_adapt_insp._memo(('joinedloader_ormadapter', self), orm_util.ORMAdapter, orm_util._TraceAdaptRole.JOINEDLOAD_MEMOIZED_ADAPTER, to_adapt_insp, equivalents=self.mapper._equivalent_columns, adapt_required=True, allow_label_resolve=False, anonymize_labels=True)
        assert clauses.is_aliased_class
        innerjoin = loadopt.local_opts.get('innerjoin', self.parent_property.innerjoin) if loadopt is not None else self.parent_property.innerjoin
        if not innerjoin:
            chained_from_outerjoin = True
        compile_state.create_eager_joins.append((self._create_eager_join, entity, path, adapter, parentmapper, clauses, innerjoin, chained_from_outerjoin, loadopt._extra_criteria if loadopt else ()))
        add_to_collection = compile_state.secondary_columns
        path.set(compile_state.attributes, 'eager_row_processor', clauses)
        return (clauses, adapter, add_to_collection, chained_from_outerjoin)

    def _create_eager_join(self, compile_state, query_entity, path, adapter, parentmapper, clauses, innerjoin, chained_from_outerjoin, extra_criteria):
        if parentmapper is None:
            localparent = query_entity.mapper
        else:
            localparent = parentmapper
        should_nest_selectable = compile_state.multi_row_eager_loaders and compile_state._should_nest_selectable
        query_entity_key = None
        if query_entity not in compile_state.eager_joins and (not should_nest_selectable) and compile_state.from_clauses:
            indexes = sql_util.find_left_clause_that_matches_given(compile_state.from_clauses, query_entity.selectable)
            if len(indexes) > 1:
                raise sa_exc.InvalidRequestError("Can't identify which query entity in which to joined eager load from.   Please use an exact match when specifying the join path.")
            if indexes:
                clause = compile_state.from_clauses[indexes[0]]
                query_entity_key, default_towrap = (indexes[0], clause)
        if query_entity_key is None:
            query_entity_key, default_towrap = (query_entity, query_entity.selectable)
        towrap = compile_state.eager_joins.setdefault(query_entity_key, default_towrap)
        if adapter:
            if getattr(adapter, 'is_aliased_class', False):
                efm = adapter.aliased_insp._entity_for_mapper(localparent if localparent.isa(self.parent) else self.parent)
                onclause = getattr(efm.entity, self.key, self.parent_property)
            else:
                onclause = getattr(orm_util.AliasedClass(self.parent, adapter.selectable, use_mapper_path=True), self.key, self.parent_property)
        else:
            onclause = self.parent_property
        assert clauses.is_aliased_class
        attach_on_outside = not chained_from_outerjoin or not innerjoin or innerjoin == 'unnested' or query_entity.entity_zero.represents_outer_join
        extra_join_criteria = extra_criteria
        additional_entity_criteria = compile_state.global_attributes.get(('additional_entity_criteria', self.mapper), ())
        if additional_entity_criteria:
            extra_join_criteria += tuple((ae._resolve_where_criteria(self.mapper) for ae in additional_entity_criteria if ae.propagate_to_loaders))
        if attach_on_outside:
            eagerjoin = orm_util._ORMJoin(towrap, clauses.aliased_insp, onclause, isouter=not innerjoin or query_entity.entity_zero.represents_outer_join or (chained_from_outerjoin and isinstance(towrap, sql.Join)), _left_memo=self.parent, _right_memo=self.mapper, _extra_criteria=extra_join_criteria)
        else:
            eagerjoin = self._splice_nested_inner_join(path, towrap, clauses, onclause, extra_join_criteria)
        compile_state.eager_joins[query_entity_key] = eagerjoin
        eagerjoin.stop_on = query_entity.selectable
        if not parentmapper:
            for col in sql_util._find_columns(self.parent_property.primaryjoin):
                if localparent.persist_selectable.c.contains_column(col):
                    if adapter:
                        col = adapter.columns[col]
                    compile_state._append_dedupe_col_collection(col, compile_state.primary_columns)
        if self.parent_property.order_by:
            compile_state.eager_order_by += tuple(eagerjoin._target_adapter.copy_and_process(util.to_list(self.parent_property.order_by)))

    def _splice_nested_inner_join(self, path, join_obj, clauses, onclause, extra_criteria, splicing=False):
        if splicing is False:
            assert isinstance(join_obj, orm_util._ORMJoin)
        elif isinstance(join_obj, sql.selectable.FromGrouping):
            return self._splice_nested_inner_join(path, join_obj.element, clauses, onclause, extra_criteria, splicing)
        elif not isinstance(join_obj, orm_util._ORMJoin):
            if path[-2].isa(splicing):
                return orm_util._ORMJoin(join_obj, clauses.aliased_insp, onclause, isouter=False, _left_memo=splicing, _right_memo=path[-1].mapper, _extra_criteria=extra_criteria)
            else:
                return None
        target_join = self._splice_nested_inner_join(path, join_obj.right, clauses, onclause, extra_criteria, join_obj._right_memo)
        if target_join is None:
            right_splice = False
            target_join = self._splice_nested_inner_join(path, join_obj.left, clauses, onclause, extra_criteria, join_obj._left_memo)
            if target_join is None:
                assert splicing is not False, 'assertion failed attempting to produce joined eager loads'
                return None
        else:
            right_splice = True
        if right_splice:
            if not join_obj.isouter and (not target_join.isouter):
                eagerjoin = join_obj._splice_into_center(target_join)
            else:
                eagerjoin = orm_util._ORMJoin(join_obj.left, target_join, join_obj.onclause, isouter=join_obj.isouter, _left_memo=join_obj._left_memo)
        else:
            eagerjoin = orm_util._ORMJoin(target_join, join_obj.right, join_obj.onclause, isouter=join_obj.isouter, _right_memo=join_obj._right_memo)
        eagerjoin._target_adapter = target_join._target_adapter
        return eagerjoin

    def _create_eager_adapter(self, context, result, adapter, path, loadopt):
        compile_state = context.compile_state
        user_defined_adapter = self._init_user_defined_eager_proc(loadopt, compile_state, context.attributes) if loadopt else False
        if user_defined_adapter is not False:
            decorator = user_defined_adapter
            if compile_state.compound_eager_adapter and decorator:
                decorator = decorator.wrap(compile_state.compound_eager_adapter)
            elif compile_state.compound_eager_adapter:
                decorator = compile_state.compound_eager_adapter
        else:
            decorator = path.get(compile_state.attributes, 'eager_row_processor')
            if decorator is None:
                return False
        if self.mapper._result_has_identity_key(result, decorator):
            return decorator
        else:
            return False

    def create_row_processor(self, context, query_entity, path, loadopt, mapper, result, adapter, populators):
        if not self.parent.class_manager[self.key].impl.supports_population:
            raise sa_exc.InvalidRequestError("'%s' does not support object population - eager loading cannot be applied." % self)
        if self.uselist:
            context.loaders_require_uniquing = True
        our_path = path[self.parent_property]
        eager_adapter = self._create_eager_adapter(context, result, adapter, our_path, loadopt)
        if eager_adapter is not False:
            key = self.key
            _instance = loading._instance_processor(query_entity, self.mapper, context, result, our_path[self.entity], eager_adapter)
            if not self.uselist:
                self._create_scalar_loader(context, key, _instance, populators)
            else:
                self._create_collection_loader(context, key, _instance, populators)
        else:
            self.parent_property._get_strategy((('lazy', 'select'),)).create_row_processor(context, query_entity, path, loadopt, mapper, result, adapter, populators)

    def _create_collection_loader(self, context, key, _instance, populators):

        def load_collection_from_joined_new_row(state, dict_, row):
            collection = attributes.init_state_collection(state, dict_, key)
            result_list = util.UniqueAppender(collection, 'append_without_event')
            context.attributes[state, key] = result_list
            inst = _instance(row)
            if inst is not None:
                result_list.append(inst)

        def load_collection_from_joined_existing_row(state, dict_, row):
            if (state, key) in context.attributes:
                result_list = context.attributes[state, key]
            else:
                collection = attributes.init_state_collection(state, dict_, key)
                result_list = util.UniqueAppender(collection, 'append_without_event')
                context.attributes[state, key] = result_list
            inst = _instance(row)
            if inst is not None:
                result_list.append(inst)

        def load_collection_from_joined_exec(state, dict_, row):
            _instance(row)
        populators['new'].append((self.key, load_collection_from_joined_new_row))
        populators['existing'].append((self.key, load_collection_from_joined_existing_row))
        if context.invoke_all_eagers:
            populators['eager'].append((self.key, load_collection_from_joined_exec))

    def _create_scalar_loader(self, context, key, _instance, populators):

        def load_scalar_from_joined_new_row(state, dict_, row):
            dict_[key] = _instance(row)

        def load_scalar_from_joined_existing_row(state, dict_, row):
            existing = _instance(row)
            if key in dict_:
                if existing is not dict_[key]:
                    util.warn("Multiple rows returned with uselist=False for eagerly-loaded attribute '%s' " % self)
            else:
                dict_[key] = existing

        def load_scalar_from_joined_exec(state, dict_, row):
            _instance(row)
        populators['new'].append((self.key, load_scalar_from_joined_new_row))
        populators['existing'].append((self.key, load_scalar_from_joined_existing_row))
        if context.invoke_all_eagers:
            populators['eager'].append((self.key, load_scalar_from_joined_exec))