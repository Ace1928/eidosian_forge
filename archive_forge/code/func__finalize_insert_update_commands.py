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
def _finalize_insert_update_commands(base_mapper, uowtransaction, states):
    """finalize state on states that have been inserted or updated,
    including calling after_insert/after_update events.

    """
    for state, state_dict, mapper, connection, has_identity in states:
        if mapper._readonly_props:
            readonly = state.unmodified_intersection([p.key for p in mapper._readonly_props if p.expire_on_flush and (not p.deferred or p.key in state.dict) or (not p.expire_on_flush and (not p.deferred) and (p.key not in state.dict))])
            if readonly:
                state._expire_attributes(state.dict, readonly)
        toload_now = []
        if base_mapper.eager_defaults is True:
            toload_now.extend(state._unloaded_non_object.intersection(mapper._server_default_plus_onupdate_propkeys))
        if mapper.version_id_col is not None and mapper.version_id_generator is False:
            if mapper._version_id_prop.key in state.unloaded:
                toload_now.extend([mapper._version_id_prop.key])
        if toload_now:
            state.key = base_mapper._identity_key_from_state(state)
            stmt = future.select(mapper).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)
            loading.load_on_ident(uowtransaction.session, stmt, state.key, refresh_state=state, only_load_props=toload_now)
        if not has_identity:
            mapper.dispatch.after_insert(mapper, connection, state)
        else:
            mapper.dispatch.after_update(mapper, connection, state)
        if mapper.version_id_generator is False and mapper.version_id_col is not None:
            if state_dict[mapper._version_id_prop.key] is None:
                raise orm_exc.FlushError('Instance does not contain a non-NULL version value')