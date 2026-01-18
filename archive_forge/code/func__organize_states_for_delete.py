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
def _organize_states_for_delete(base_mapper, states, uowtransaction):
    """Make an initial pass across a set of states for DELETE.

    This includes calling out before_delete and obtaining
    key information for each state including its dictionary,
    mapper, the connection to use for the execution per state.

    """
    for state, dict_, mapper, connection in _connections_for_states(base_mapper, uowtransaction, states):
        mapper.dispatch.before_delete(mapper, connection, state)
        if mapper.version_id_col is not None:
            update_version_id = mapper._get_committed_state_attr_by_column(state, dict_, mapper.version_id_col)
        else:
            update_version_id = None
        yield (state, dict_, mapper, connection, update_version_id)