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