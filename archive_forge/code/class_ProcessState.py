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
class ProcessState(PostSortRec):
    __slots__ = ('dependency_processor', 'isdelete', 'state', 'sort_key')

    def __init__(self, uow, dependency_processor, isdelete, state):
        self.dependency_processor = dependency_processor
        self.sort_key = ('ProcessState', dependency_processor.sort_key)
        self.isdelete = isdelete
        self.state = state

    def execute_aggregate(self, uow, recs):
        cls_ = self.__class__
        dependency_processor = self.dependency_processor
        isdelete = self.isdelete
        our_recs = [r for r in recs if r.__class__ is cls_ and r.dependency_processor is dependency_processor and (r.isdelete is isdelete)]
        recs.difference_update(our_recs)
        states = [self.state] + [r.state for r in our_recs]
        if isdelete:
            dependency_processor.process_deletes(uow, states)
        else:
            dependency_processor.process_saves(uow, states)

    def __repr__(self):
        return '%s(%s, %s, delete=%s)' % (self.__class__.__name__, self.dependency_processor, orm_util.state_str(self.state), self.isdelete)