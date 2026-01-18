from __future__ import annotations
from itertools import chain
from itertools import groupby
from itertools import zip_longest
import operator
from . import attributes
from . import exc as orm_exc
from . import loading
from . import sync
from .base import state_str
from .. import exc as sa_exc
from .. import future
from .. import sql
from .. import util
from ..engine import cursor as _cursor
from ..sql import operators
from ..sql.elements import BooleanClauseList
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
def _postfetch_post_update(mapper, uowtransaction, table, state, dict_, result, params):
    needs_version_id = mapper.version_id_col is not None and mapper.version_id_col in mapper._cols_by_table[table]
    if not uowtransaction.is_deleted(state):
        prefetch_cols = result.context.compiled.prefetch
        postfetch_cols = result.context.compiled.postfetch
    elif needs_version_id:
        prefetch_cols = postfetch_cols = ()
    else:
        return
    if needs_version_id:
        prefetch_cols = list(prefetch_cols) + [mapper.version_id_col]
    refresh_flush = bool(mapper.class_manager.dispatch.refresh_flush)
    if refresh_flush:
        load_evt_attrs = []
    for c in prefetch_cols:
        if c.key in params and c in mapper._columntoproperty:
            dict_[mapper._columntoproperty[c].key] = params[c.key]
            if refresh_flush:
                load_evt_attrs.append(mapper._columntoproperty[c].key)
    if refresh_flush and load_evt_attrs:
        mapper.class_manager.dispatch.refresh_flush(state, uowtransaction, load_evt_attrs)
    if postfetch_cols:
        state._expire_attributes(state.dict, [mapper._columntoproperty[c].key for c in postfetch_cols if c in mapper._columntoproperty])