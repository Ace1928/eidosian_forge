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
def _organize_states_for_post_update(base_mapper, states, uowtransaction):
    """Make an initial pass across a set of states for UPDATE
    corresponding to post_update.

    This includes obtaining key information for each state
    including its dictionary, mapper, the connection to use for
    the execution per state.

    """
    return _connections_for_states(base_mapper, uowtransaction, states)