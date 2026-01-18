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
@relationships.RelationshipProperty.strategy_for(lazy='selectin')
class SelectInLoader(PostLoader, util.MemoizedSlots):
    __slots__ = ('join_depth', 'omit_join', '_parent_alias', '_query_info', '_fallback_query_info')
    query_info = collections.namedtuple('queryinfo', ['load_only_child', 'load_with_join', 'in_expr', 'pk_cols', 'zero_idx', 'child_lookup_cols'])
    _chunksize = 500

    def __init__(self, parent, strategy_key):
        super().__init__(parent, strategy_key)
        self.join_depth = self.parent_property.join_depth
        is_m2o = self.parent_property.direction is interfaces.MANYTOONE
        if self.parent_property.omit_join is not None:
            self.omit_join = self.parent_property.omit_join
        else:
            lazyloader = self.parent_property._get_strategy((('lazy', 'select'),))
            if is_m2o:
                self.omit_join = lazyloader.use_get
            else:
                self.omit_join = self.parent._get_clause[0].compare(lazyloader._rev_lazywhere, use_proxies=True, compare_keys=False, equivalents=self.parent._equivalent_columns)
        if self.omit_join:
            if is_m2o:
                self._query_info = self._init_for_omit_join_m2o()
                self._fallback_query_info = self._init_for_join()
            else:
                self._query_info = self._init_for_omit_join()
        else:
            self._query_info = self._init_for_join()

    def _init_for_omit_join(self):
        pk_to_fk = dict(self.parent_property._join_condition.local_remote_pairs)
        pk_to_fk.update(((equiv, pk_to_fk[k]) for k in list(pk_to_fk) for equiv in self.parent._equivalent_columns.get(k, ())))
        pk_cols = fk_cols = [pk_to_fk[col] for col in self.parent.primary_key if col in pk_to_fk]
        if len(fk_cols) > 1:
            in_expr = sql.tuple_(*fk_cols)
            zero_idx = False
        else:
            in_expr = fk_cols[0]
            zero_idx = True
        return self.query_info(False, False, in_expr, pk_cols, zero_idx, None)

    def _init_for_omit_join_m2o(self):
        pk_cols = self.mapper.primary_key
        if len(pk_cols) > 1:
            in_expr = sql.tuple_(*pk_cols)
            zero_idx = False
        else:
            in_expr = pk_cols[0]
            zero_idx = True
        lazyloader = self.parent_property._get_strategy((('lazy', 'select'),))
        lookup_cols = [lazyloader._equated_columns[pk] for pk in pk_cols]
        return self.query_info(True, False, in_expr, pk_cols, zero_idx, lookup_cols)

    def _init_for_join(self):
        self._parent_alias = AliasedClass(self.parent.class_)
        pa_insp = inspect(self._parent_alias)
        pk_cols = [pa_insp._adapt_element(col) for col in self.parent.primary_key]
        if len(pk_cols) > 1:
            in_expr = sql.tuple_(*pk_cols)
            zero_idx = False
        else:
            in_expr = pk_cols[0]
            zero_idx = True
        return self.query_info(False, True, in_expr, pk_cols, zero_idx, None)

    def init_class_attribute(self, mapper):
        self.parent_property._get_strategy((('lazy', 'select'),)).init_class_attribute(mapper)

    def create_row_processor(self, context, query_entity, path, loadopt, mapper, result, adapter, populators):
        if context.refresh_state:
            return self._immediateload_create_row_processor(context, query_entity, path, loadopt, mapper, result, adapter, populators)
        effective_path, run_loader, execution_options, recursion_depth = self._setup_for_recursion(context, path, loadopt, join_depth=self.join_depth)
        if not run_loader:
            return
        if not self.parent.class_manager[self.key].impl.supports_population:
            raise sa_exc.InvalidRequestError("'%s' does not support object population - eager loading cannot be applied." % self)
        if len(path) == 1:
            if not orm_util._entity_isa(query_entity.entity_zero, self.parent):
                return
        elif not orm_util._entity_isa(path[-1], self.parent):
            return
        selectin_path = effective_path
        path_w_prop = path[self.parent_property]
        with_poly_entity = path_w_prop.get(context.attributes, 'path_with_polymorphic', None)
        if with_poly_entity is not None:
            effective_entity = inspect(with_poly_entity)
        else:
            effective_entity = self.entity
        loading.PostLoad.callable_for_path(context, selectin_path, self.parent, self.parent_property, self._load_for_path, effective_entity, loadopt, recursion_depth, execution_options)

    def _load_for_path(self, context, path, states, load_only, effective_entity, loadopt, recursion_depth, execution_options):
        if load_only and self.key not in load_only:
            return
        query_info = self._query_info
        if query_info.load_only_child:
            our_states = collections.defaultdict(list)
            none_states = []
            mapper = self.parent
            for state, overwrite in states:
                state_dict = state.dict
                related_ident = tuple((mapper._get_state_attr_by_column(state, state_dict, lk, passive=attributes.PASSIVE_NO_FETCH) for lk in query_info.child_lookup_cols))
                if LoaderCallableStatus.PASSIVE_NO_RESULT in related_ident:
                    query_info = self._fallback_query_info
                    break
                if None not in related_ident:
                    our_states[related_ident].append((state, state_dict, overwrite))
                else:
                    none_states.append((state, state_dict, overwrite))
        if not query_info.load_only_child:
            our_states = [(state.key[1], state, state.dict, overwrite) for state, overwrite in states]
        pk_cols = query_info.pk_cols
        in_expr = query_info.in_expr
        if not query_info.load_with_join:
            if effective_entity.is_aliased_class:
                pk_cols = [effective_entity._adapt_element(col) for col in pk_cols]
                in_expr = effective_entity._adapt_element(in_expr)
        bundle_ent = orm_util.Bundle('pk', *pk_cols)
        bundle_sql = bundle_ent.__clause_element__()
        entity_sql = effective_entity.__clause_element__()
        q = Select._create_raw_select(_raw_columns=[bundle_sql, entity_sql], _label_style=LABEL_STYLE_TABLENAME_PLUS_COL, _compile_options=ORMCompileState.default_compile_options, _propagate_attrs={'compile_state_plugin': 'orm', 'plugin_subject': effective_entity})
        if not query_info.load_with_join:
            q = q.select_from(effective_entity)
        else:
            q = q.select_from(self._parent_alias).join(getattr(self._parent_alias, self.parent_property.key).of_type(effective_entity))
        q = q.filter(in_expr.in_(sql.bindparam('primary_keys')))
        orig_query = context.compile_state.select_statement
        effective_path = path[self.parent_property]
        if orig_query is context.query:
            new_options = orig_query._with_options
        else:
            cached_options = orig_query._with_options
            uncached_options = context.query._with_options
            new_options = [orig_opt._adapt_cached_option_to_uncached_option(context, uncached_opt) for orig_opt, uncached_opt in zip(cached_options, uncached_options)]
        if loadopt and loadopt._extra_criteria:
            new_options += (orm_util.LoaderCriteriaOption(effective_entity, loadopt._generate_extra_criteria(context)),)
        if recursion_depth is not None:
            effective_path = effective_path._truncate_recursive()
        q = q.options(*new_options)
        q = q._update_compile_options({'_current_path': effective_path})
        if context.populate_existing:
            q = q.execution_options(populate_existing=True)
        if self.parent_property.order_by:
            if not query_info.load_with_join:
                eager_order_by = self.parent_property.order_by
                if effective_entity.is_aliased_class:
                    eager_order_by = [effective_entity._adapt_element(elem) for elem in eager_order_by]
                q = q.order_by(*eager_order_by)
            else:

                def _setup_outermost_orderby(compile_context):
                    compile_context.eager_order_by += tuple(util.to_list(self.parent_property.order_by))
                q = q._add_context_option(_setup_outermost_orderby, self.parent_property)
        if query_info.load_only_child:
            self._load_via_child(our_states, none_states, query_info, q, context, execution_options)
        else:
            self._load_via_parent(our_states, query_info, q, context, execution_options)

    def _load_via_child(self, our_states, none_states, query_info, q, context, execution_options):
        uselist = self.uselist
        our_keys = sorted(our_states)
        while our_keys:
            chunk = our_keys[0:self._chunksize]
            our_keys = our_keys[self._chunksize:]
            data = {k: v for k, v in context.session.execute(q, params={'primary_keys': [key[0] if query_info.zero_idx else key for key in chunk]}, execution_options=execution_options).unique()}
            for key in chunk:
                related_obj = data.get(key, None)
                for state, dict_, overwrite in our_states[key]:
                    if not overwrite and self.key in dict_:
                        continue
                    state.get_impl(self.key).set_committed_value(state, dict_, related_obj if not uselist else [related_obj])
        for state, dict_, overwrite in none_states:
            if not overwrite and self.key in dict_:
                continue
            state.get_impl(self.key).set_committed_value(state, dict_, None)

    def _load_via_parent(self, our_states, query_info, q, context, execution_options):
        uselist = self.uselist
        _empty_result = () if uselist else None
        while our_states:
            chunk = our_states[0:self._chunksize]
            our_states = our_states[self._chunksize:]
            primary_keys = [key[0] if query_info.zero_idx else key for key, state, state_dict, overwrite in chunk]
            data = collections.defaultdict(list)
            for k, v in itertools.groupby(context.session.execute(q, params={'primary_keys': primary_keys}, execution_options=execution_options).unique(), lambda x: x[0]):
                data[k].extend((vv[1] for vv in v))
            for key, state, state_dict, overwrite in chunk:
                if not overwrite and self.key in state_dict:
                    continue
                collection = data.get(key, _empty_result)
                if not uselist and collection:
                    if len(collection) > 1:
                        util.warn("Multiple rows returned with uselist=False for eagerly-loaded attribute '%s' " % self)
                    state.get_impl(self.key).set_committed_value(state, state_dict, collection[0])
                else:
                    state.get_impl(self.key).set_committed_value(state, state_dict, collection)