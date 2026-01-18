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
@util.preload_module('sqlalchemy.orm.strategy_options')
def _emit_lazyload(self, session, state, primary_key_identity, passive, loadopt, extra_criteria, extra_options, alternate_effective_path, execution_options):
    strategy_options = util.preloaded.orm_strategy_options
    clauseelement = self.entity.__clause_element__()
    stmt = Select._create_raw_select(_raw_columns=[clauseelement], _propagate_attrs=clauseelement._propagate_attrs, _label_style=LABEL_STYLE_TABLENAME_PLUS_COL, _compile_options=ORMCompileState.default_compile_options)
    load_options = QueryContext.default_load_options
    load_options += {'_invoke_all_eagers': False, '_lazy_loaded_from': state}
    if self.parent_property.secondary is not None:
        stmt = stmt.select_from(self.mapper, self.parent_property.secondary)
    pending = not state.key
    if pending or passive & attributes.NO_AUTOFLUSH:
        stmt._execution_options = util.immutabledict({'autoflush': False})
    use_get = self.use_get
    if state.load_options or (loadopt and loadopt._extra_criteria):
        if alternate_effective_path is None:
            effective_path = state.load_path[self.parent_property]
        else:
            effective_path = alternate_effective_path[self.parent_property]
        opts = state.load_options
        if loadopt and loadopt._extra_criteria:
            use_get = False
            opts += (orm_util.LoaderCriteriaOption(self.entity, extra_criteria),)
        stmt._with_options = opts
    elif alternate_effective_path is None:
        effective_path = state.mapper._path_registry[self.parent_property]
    else:
        effective_path = alternate_effective_path[self.parent_property]
    if extra_options:
        stmt._with_options += extra_options
    stmt._compile_options += {'_current_path': effective_path}
    if use_get:
        if self._raise_on_sql and (not passive & PassiveFlag.NO_RAISE):
            self._invoke_raise_load(state, passive, 'raise_on_sql')
        return loading.load_on_pk_identity(session, stmt, primary_key_identity, load_options=load_options, execution_options=execution_options)
    if self._order_by:
        stmt._order_by_clauses = self._order_by

    def _lazyload_reverse(compile_context):
        for rev in self.parent_property._reverse_property:
            if rev.direction is interfaces.MANYTOONE and rev._use_get and (not isinstance(rev.strategy, LazyLoader)):
                strategy_options.Load._construct_for_existing_path(compile_context.compile_options._current_path[rev.parent]).lazyload(rev).process_compile_state(compile_context)
    stmt._with_context_options += ((_lazyload_reverse, self.parent_property),)
    lazy_clause, params = self._generate_lazy_clause(state, passive)
    if execution_options:
        execution_options = util.EMPTY_DICT.merge_with(execution_options, {'_sa_orm_load_options': load_options})
    else:
        execution_options = {'_sa_orm_load_options': load_options}
    if self.key in state.dict and (not passive & PassiveFlag.DEFERRED_HISTORY_LOAD):
        return LoaderCallableStatus.ATTR_WAS_SET
    if pending:
        if util.has_intersection(orm_util._none_set, params.values()):
            return None
    elif util.has_intersection(orm_util._never_set, params.values()):
        return None
    if self._raise_on_sql and (not passive & PassiveFlag.NO_RAISE):
        self._invoke_raise_load(state, passive, 'raise_on_sql')
    stmt._where_criteria = (lazy_clause,)
    result = session.execute(stmt, params, execution_options=execution_options)
    result = result.unique().scalars().all()
    if self.uselist:
        return result
    else:
        l = len(result)
        if l:
            if l > 1:
                util.warn("Multiple rows returned with uselist=False for lazily-loaded attribute '%s' " % self.parent_property)
            return result[0]
        else:
            return None