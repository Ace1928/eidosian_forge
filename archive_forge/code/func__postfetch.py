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
def _postfetch(mapper, uowtransaction, table, state, dict_, result, params, value_params, isupdate, returned_defaults):
    """Expire attributes in need of newly persisted database state,
    after an INSERT or UPDATE statement has proceeded for that
    state."""
    prefetch_cols = result.context.compiled.prefetch
    postfetch_cols = result.context.compiled.postfetch
    returning_cols = result.context.compiled.effective_returning
    if mapper.version_id_col is not None and mapper.version_id_col in mapper._cols_by_table[table]:
        prefetch_cols = list(prefetch_cols) + [mapper.version_id_col]
    refresh_flush = bool(mapper.class_manager.dispatch.refresh_flush)
    if refresh_flush:
        load_evt_attrs = []
    if returning_cols:
        row = returned_defaults
        if row is not None:
            for row_value, col in zip(row, returning_cols):
                if col.primary_key and result.context.isinsert:
                    continue
                prop = mapper._columntoproperty.get(col)
                if prop:
                    dict_[prop.key] = row_value
                    if refresh_flush:
                        load_evt_attrs.append(prop.key)
    for c in prefetch_cols:
        if c.key in params and c in mapper._columntoproperty:
            pkey = mapper._columntoproperty[c].key
            dict_[pkey] = params[c.key]
            state.committed_state.pop(pkey, None)
            if refresh_flush:
                load_evt_attrs.append(pkey)
    if refresh_flush and load_evt_attrs:
        mapper.class_manager.dispatch.refresh_flush(state, uowtransaction, load_evt_attrs)
    if isupdate and value_params:
        postfetch_cols.extend([col for col in value_params if col.primary_key and col not in returning_cols])
    if postfetch_cols:
        state._expire_attributes(state.dict, [mapper._columntoproperty[c].key for c in postfetch_cols if c in mapper._columntoproperty])
    for m, equated_pairs in mapper._table_to_equated[table]:
        sync.populate(state, m, state, m, equated_pairs, uowtransaction, mapper.passive_updates)