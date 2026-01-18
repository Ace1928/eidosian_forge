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