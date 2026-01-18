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
class SaveUpdateState(PostSortRec):
    __slots__ = ('state', 'mapper', 'sort_key')

    def __init__(self, uow, state):
        self.state = state
        self.mapper = state.mapper.base_mapper
        self.sort_key = ('ProcessState', self.mapper._sort_key)

    @util.preload_module('sqlalchemy.orm.persistence')
    def execute_aggregate(self, uow, recs):
        persistence = util.preloaded.orm_persistence
        cls_ = self.__class__
        mapper = self.mapper
        our_recs = [r for r in recs if r.__class__ is cls_ and r.mapper is mapper]
        recs.difference_update(our_recs)
        persistence.save_obj(mapper, [self.state] + [r.state for r in our_recs], uow)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, orm_util.state_str(self.state))