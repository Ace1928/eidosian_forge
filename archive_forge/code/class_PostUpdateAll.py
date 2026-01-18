from __future__ import annotations
from typing import Any
from typing import Dict
from typing import Optional
from typing import Set
from typing import TYPE_CHECKING
from . import attributes
from . import exc as orm_exc
from . import util as orm_util
from .. import event
from .. import util
from ..util import topological
class PostUpdateAll(PostSortRec):
    __slots__ = ('mapper', 'isdelete', 'sort_key')

    def __init__(self, uow, mapper, isdelete):
        self.mapper = mapper
        self.isdelete = isdelete
        self.sort_key = ('PostUpdateAll', mapper._sort_key, isdelete)

    @util.preload_module('sqlalchemy.orm.persistence')
    def execute(self, uow):
        persistence = util.preloaded.orm_persistence
        states, cols = uow.post_update_states[self.mapper]
        states = [s for s in states if uow.states[s][0] == self.isdelete]
        persistence.post_update(self.mapper, states, uow, cols)