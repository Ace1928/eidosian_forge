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
def _generate_from_original_query(self, orig_compile_state, orig_query, leftmost_mapper, leftmost_attr, leftmost_relationship, orig_entity):
    q = orig_query._clone().correlate(None)
    q2 = query.Query.__new__(query.Query)
    q2.__dict__.update(q.__dict__)
    q = q2
    if not q._from_obj:
        q._enable_assertions = False
        q.select_from.non_generative(q, *{ent['entity'] for ent in _column_descriptions(orig_query, compile_state=orig_compile_state) if ent['entity'] is not None})
    target_cols = orig_compile_state._adapt_col_list([sql.coercions.expect(sql.roles.ColumnsClauseRole, o) for o in leftmost_attr], orig_compile_state._get_current_adapter())
    q._raw_columns = target_cols
    distinct_target_key = leftmost_relationship.distinct_target_key
    if distinct_target_key is True:
        q._distinct = True
    elif distinct_target_key is None:
        for t in {c.table for c in target_cols}:
            if not set(target_cols).issuperset(t.primary_key):
                q._distinct = True
                break
    if not q._has_row_limiting_clause:
        q._order_by_clauses = ()
    if q._distinct is True and q._order_by_clauses:
        to_add = sql_util.expand_column_list_from_order_by(target_cols, q._order_by_clauses)
        if to_add:
            q._set_entities(target_cols + to_add)
    embed_q = q.set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL).subquery()
    left_alias = orm_util.AliasedClass(leftmost_mapper, embed_q, use_mapper_path=True)
    return left_alias