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
class PostLoad:
    """Track loaders and states for "post load" operations."""
    __slots__ = ('loaders', 'states', 'load_keys')

    def __init__(self):
        self.loaders = {}
        self.states = util.OrderedDict()
        self.load_keys = None

    def add_state(self, state, overwrite):
        self.states[state] = overwrite

    def invoke(self, context, path):
        if not self.states:
            return
        path = path_registry.PathRegistry.coerce(path)
        for effective_context, token, limit_to_mapper, loader, arg, kw in self.loaders.values():
            states = [(state, overwrite) for state, overwrite in self.states.items() if state.manager.mapper.isa(limit_to_mapper)]
            if states:
                loader(effective_context, path, states, self.load_keys, *arg, **kw)
        self.states.clear()

    @classmethod
    def for_context(cls, context, path, only_load_props):
        pl = context.post_load_paths.get(path.path)
        if pl is not None and only_load_props:
            pl.load_keys = only_load_props
        return pl

    @classmethod
    def path_exists(self, context, path, key):
        return path.path in context.post_load_paths and key in context.post_load_paths[path.path].loaders

    @classmethod
    def callable_for_path(cls, context, path, limit_to_mapper, token, loader_callable, *arg, **kw):
        if path.path in context.post_load_paths:
            pl = context.post_load_paths[path.path]
        else:
            pl = context.post_load_paths[path.path] = PostLoad()
        pl.loaders[token] = (context, token, limit_to_mapper, loader_callable, arg, kw)