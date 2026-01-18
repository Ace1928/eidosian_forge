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
class Preprocess(IterateMappersMixin):
    __slots__ = ('dependency_processor', 'fromparent', 'processed', 'setup_flush_actions')

    def __init__(self, dependency_processor, fromparent):
        self.dependency_processor = dependency_processor
        self.fromparent = fromparent
        self.processed = set()
        self.setup_flush_actions = False

    def execute(self, uow):
        delete_states = set()
        save_states = set()
        for mapper in self._mappers(uow):
            for state in uow.mappers[mapper].difference(self.processed):
                isdelete, listonly = uow.states[state]
                if not listonly:
                    if isdelete:
                        delete_states.add(state)
                    else:
                        save_states.add(state)
        if delete_states:
            self.dependency_processor.presort_deletes(uow, delete_states)
            self.processed.update(delete_states)
        if save_states:
            self.dependency_processor.presort_saves(uow, save_states)
            self.processed.update(save_states)
        if delete_states or save_states:
            if not self.setup_flush_actions and (self.dependency_processor.prop_has_changes(uow, delete_states, True) or self.dependency_processor.prop_has_changes(uow, save_states, False)):
                self.dependency_processor.per_property_flush_actions(uow)
                self.setup_flush_actions = True
            return True
        else:
            return False