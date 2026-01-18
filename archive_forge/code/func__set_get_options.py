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
def _set_get_options(compile_opt, load_opt, populate_existing=None, version_check=None, only_load_props=None, refresh_state=None, identity_token=None, is_user_refresh=None):
    compile_options = {}
    load_options = {}
    if version_check:
        load_options['_version_check'] = version_check
    if populate_existing:
        load_options['_populate_existing'] = populate_existing
    if refresh_state:
        load_options['_refresh_state'] = refresh_state
        compile_options['_for_refresh_state'] = True
    if only_load_props:
        compile_options['_only_load_props'] = frozenset(only_load_props)
    if identity_token:
        load_options['_identity_token'] = identity_token
    if is_user_refresh:
        load_options['_is_user_refresh'] = is_user_refresh
    if load_options:
        load_opt += load_options
    if compile_options:
        compile_opt += compile_options
    return (compile_opt, load_opt)