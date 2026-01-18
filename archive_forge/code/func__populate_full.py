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
def _populate_full(context, row, state, dict_, isnew, load_path, loaded_instance, populate_existing, populators):
    if isnew:
        state.runid = context.runid
        for key, getter in populators['quick']:
            dict_[key] = getter(row)
        if populate_existing:
            for key, set_callable in populators['expire']:
                dict_.pop(key, None)
                if set_callable:
                    state.expired_attributes.add(key)
        else:
            for key, set_callable in populators['expire']:
                if set_callable:
                    state.expired_attributes.add(key)
        for key, populator in populators['new']:
            populator(state, dict_, row)
    elif load_path != state.load_path:
        state.load_path = load_path
        for key, getter in populators['quick']:
            if key not in dict_:
                dict_[key] = getter(row)
        for key, populator in populators['existing']:
            populator(state, dict_, row)
    else:
        for key, populator in populators['existing']:
            populator(state, dict_, row)