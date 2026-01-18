from __future__ import annotations
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import exc as orm_exc
from . import path_registry
from .base import _DEFER_FOR_STATE
from .base import _RAISE_FOR_STATE
from .base import _SET_DEFERRED_EXPIRED
from .base import PassiveFlag
from .context import FromStatement
from .context import ORMCompileState
from .context import QueryContext
from .util import _none_set
from .util import state_str
from .. import exc as sa_exc
from .. import util
from ..engine import result_tuple
from ..engine.result import ChunkedIteratorResult
from ..engine.result import FrozenResult
from ..engine.result import SimpleResultMetaData
from ..sql import select
from ..sql import util as sql_util
from ..sql.selectable import ForUpdateArg
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import SelectState
from ..util import EMPTY_DICT
def _populate_partial(context, row, state, dict_, isnew, load_path, unloaded, populators):
    if not isnew:
        if unloaded:
            for key, getter in populators['quick']:
                if key in unloaded:
                    dict_[key] = getter(row)
        to_load = context.partials[state]
        for key, populator in populators['existing']:
            if key in to_load:
                populator(state, dict_, row)
    else:
        to_load = unloaded
        context.partials[state] = to_load
        for key, getter in populators['quick']:
            if key in to_load:
                dict_[key] = getter(row)
        for key, set_callable in populators['expire']:
            if key in to_load:
                dict_.pop(key, None)
                if set_callable:
                    state.expired_attributes.add(key)
        for key, populator in populators['new']:
            if key in to_load:
                populator(state, dict_, row)
    for key, populator in populators['eager']:
        if key not in unloaded:
            populator(state, dict_, row)
    return to_load