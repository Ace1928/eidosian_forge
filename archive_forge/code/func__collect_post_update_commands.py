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
def _collect_post_update_commands(base_mapper, uowtransaction, table, states_to_update, post_update_cols):
    """Identify sets of values to use in UPDATE statements for a
    list of states within a post_update operation.

    """
    for state, state_dict, mapper, connection, update_version_id in states_to_update:
        pks = mapper._pks_by_table[table]
        params = {}
        hasdata = False
        for col in mapper._cols_by_table[table]:
            if col in pks:
                params[col._label] = mapper._get_state_attr_by_column(state, state_dict, col, passive=attributes.PASSIVE_OFF)
            elif col in post_update_cols or col.onupdate is not None:
                prop = mapper._columntoproperty[col]
                history = state.manager[prop.key].impl.get_history(state, state_dict, attributes.PASSIVE_NO_INITIALIZE)
                if history.added:
                    value = history.added[0]
                    params[col.key] = value
                    hasdata = True
        if hasdata:
            if update_version_id is not None and mapper.version_id_col in mapper._cols_by_table[table]:
                col = mapper.version_id_col
                params[col._label] = update_version_id
                if bool(state.key) and col.key not in params and (mapper.version_id_generator is not False):
                    val = mapper.version_id_generator(update_version_id)
                    params[col.key] = val
            yield (state, state_dict, mapper, connection, params)