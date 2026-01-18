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
class ProcessAll(IterateMappersMixin, PostSortRec):
    __slots__ = ('dependency_processor', 'isdelete', 'fromparent', 'sort_key')

    def __init__(self, uow, dependency_processor, isdelete, fromparent):
        self.dependency_processor = dependency_processor
        self.sort_key = ('ProcessAll', self.dependency_processor.sort_key, isdelete)
        self.isdelete = isdelete
        self.fromparent = fromparent
        uow.deps[dependency_processor.parent.base_mapper].add(dependency_processor)

    def execute(self, uow):
        states = self._elements(uow)
        if self.isdelete:
            self.dependency_processor.process_deletes(uow, states)
        else:
            self.dependency_processor.process_saves(uow, states)

    def per_state_flush_actions(self, uow):
        return iter([])

    def __repr__(self):
        return '%s(%s, isdelete=%s)' % (self.__class__.__name__, self.dependency_processor, self.isdelete)

    def _elements(self, uow):
        for mapper in self._mappers(uow):
            for state in uow.mappers[mapper]:
                isdelete, listonly = uow.states[state]
                if isdelete == self.isdelete and (not listonly):
                    yield state