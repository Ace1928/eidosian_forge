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
class SaveUpdateAll(PostSortRec):
    __slots__ = ('mapper', 'sort_key')

    def __init__(self, uow, mapper):
        self.mapper = mapper
        self.sort_key = ('SaveUpdateAll', mapper._sort_key)
        assert mapper is mapper.base_mapper

    @util.preload_module('sqlalchemy.orm.persistence')
    def execute(self, uow):
        util.preloaded.orm_persistence.save_obj(self.mapper, uow.states_for_mapper_hierarchy(self.mapper, False, False), uow)

    def per_state_flush_actions(self, uow):
        states = list(uow.states_for_mapper_hierarchy(self.mapper, False, False))
        base_mapper = self.mapper.base_mapper
        delete_all = DeleteAll(uow, base_mapper)
        for state in states:
            action = SaveUpdateState(uow, state)
            uow.dependencies.add((action, delete_all))
            yield action
        for dep in uow.deps[self.mapper]:
            states_for_prop = uow.filter_states_for_dep(dep, states)
            dep.per_state_flush_actions(uow, states_for_prop, False)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.mapper)