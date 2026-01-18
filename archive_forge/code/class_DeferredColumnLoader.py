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
@properties.ColumnProperty.strategy_for(deferred=True, instrument=True)
@properties.ColumnProperty.strategy_for(deferred=True, instrument=True, raiseload=True)
@properties.ColumnProperty.strategy_for(do_nothing=True)
class DeferredColumnLoader(LoaderStrategy):
    """Provide loading behavior for a deferred :class:`.ColumnProperty`."""
    __slots__ = ('columns', 'group', 'raiseload')

    def __init__(self, parent, strategy_key):
        super().__init__(parent, strategy_key)
        if hasattr(self.parent_property, 'composite_class'):
            raise NotImplementedError('Deferred loading for composite types not implemented yet')
        self.raiseload = self.strategy_opts.get('raiseload', False)
        self.columns = self.parent_property.columns
        self.group = self.parent_property.group

    def create_row_processor(self, context, query_entity, path, loadopt, mapper, result, adapter, populators):
        if context.refresh_state and context.query._compile_options._only_load_props and (self.key in context.query._compile_options._only_load_props):
            self.parent_property._get_strategy((('deferred', False), ('instrument', True))).create_row_processor(context, query_entity, path, loadopt, mapper, result, adapter, populators)
        elif not self.is_class_level:
            if self.raiseload:
                set_deferred_for_local_state = self.parent_property._raise_column_loader
            else:
                set_deferred_for_local_state = self.parent_property._deferred_column_loader
            populators['new'].append((self.key, set_deferred_for_local_state))
        else:
            populators['expire'].append((self.key, False))

    def init_class_attribute(self, mapper):
        self.is_class_level = True
        _register_attribute(self.parent_property, mapper, useobject=False, compare_function=self.columns[0].type.compare_values, callable_=self._load_for_state, load_on_unexpire=False)

    def setup_query(self, compile_state, query_entity, path, loadopt, adapter, column_collection, memoized_populators, only_load_props=None, **kw):
        if compile_state.compile_options._render_for_subquery and self.parent_property._renders_in_subqueries or (loadopt and set(self.columns).intersection(self.parent._should_undefer_in_wildcard)) or (loadopt and self.group and loadopt.local_opts.get('undefer_group_%s' % self.group, False)) or (only_load_props and self.key in only_load_props):
            self.parent_property._get_strategy((('deferred', False), ('instrument', True))).setup_query(compile_state, query_entity, path, loadopt, adapter, column_collection, memoized_populators, **kw)
        elif self.is_class_level:
            memoized_populators[self.parent_property] = _SET_DEFERRED_EXPIRED
        elif not self.raiseload:
            memoized_populators[self.parent_property] = _DEFER_FOR_STATE
        else:
            memoized_populators[self.parent_property] = _RAISE_FOR_STATE

    def _load_for_state(self, state, passive):
        if not state.key:
            return LoaderCallableStatus.ATTR_EMPTY
        if not passive & PassiveFlag.SQL_OK:
            return LoaderCallableStatus.PASSIVE_NO_RESULT
        localparent = state.manager.mapper
        if self.group:
            toload = [p.key for p in localparent.iterate_properties if isinstance(p, StrategizedProperty) and isinstance(p.strategy, DeferredColumnLoader) and (p.group == self.group)]
        else:
            toload = [self.key]
        group = [k for k in toload if k in state.unmodified]
        session = _state_session(state)
        if session is None:
            raise orm_exc.DetachedInstanceError("Parent instance %s is not bound to a Session; deferred load operation of attribute '%s' cannot proceed" % (orm_util.state_str(state), self.key))
        if self.raiseload:
            self._invoke_raise_load(state, passive, 'raise')
        loading.load_scalar_attributes(state.mapper, state, set(group), PASSIVE_OFF)
        return LoaderCallableStatus.ATTR_WAS_SET

    def _invoke_raise_load(self, state, passive, lazy):
        raise sa_exc.InvalidRequestError("'%s' is not available due to raiseload=True" % (self,))