from __future__ import annotations
from . import attributes
from . import exc
from . import sync
from . import unitofwork
from . import util as mapperutil
from .interfaces import MANYTOMANY
from .interfaces import MANYTOONE
from .interfaces import ONETOMANY
from .. import exc as sa_exc
from .. import sql
from .. import util
class DetectKeySwitch(DependencyProcessor):
    """For many-to-one relationships with no one-to-many backref,
    searches for parents through the unit of work when a primary
    key has changed and updates them.

    Theoretically, this approach could be expanded to support transparent
    deletion of objects referenced via many-to-one as well, although
    the current attribute system doesn't do enough bookkeeping for this
    to be efficient.

    """

    def per_property_preprocessors(self, uow):
        if self.prop._reverse_property:
            if self.passive_updates:
                return
            elif False in (prop.passive_updates for prop in self.prop._reverse_property):
                return
        uow.register_preprocessor(self, False)

    def per_property_flush_actions(self, uow):
        parent_saves = unitofwork.SaveUpdateAll(uow, self.parent.base_mapper)
        after_save = unitofwork.ProcessAll(uow, self, False, False)
        uow.dependencies.update([(parent_saves, after_save)])

    def per_state_flush_actions(self, uow, states, isdelete):
        pass

    def presort_deletes(self, uowcommit, states):
        pass

    def presort_saves(self, uow, states):
        if not self.passive_updates:
            self._process_key_switches(states, uow)

    def prop_has_changes(self, uow, states, isdelete):
        if not isdelete and self.passive_updates:
            d = self._key_switchers(uow, states)
            return bool(d)
        return False

    def process_deletes(self, uowcommit, states):
        assert False

    def process_saves(self, uowcommit, states):
        assert self.passive_updates
        self._process_key_switches(states, uowcommit)

    def _key_switchers(self, uow, states):
        switched, notswitched = uow.memo(('pk_switchers', self), lambda: (set(), set()))
        allstates = switched.union(notswitched)
        for s in states:
            if s not in allstates:
                if self._pks_changed(uow, s):
                    switched.add(s)
                else:
                    notswitched.add(s)
        return switched

    def _process_key_switches(self, deplist, uowcommit):
        switchers = self._key_switchers(uowcommit, deplist)
        if switchers:
            for state in uowcommit.session.identity_map.all_states():
                if not issubclass(state.class_, self.parent.class_):
                    continue
                dict_ = state.dict
                related = state.get_impl(self.key).get(state, dict_, passive=self._passive_update_flag)
                if related is not attributes.PASSIVE_NO_RESULT and related is not None:
                    if self.prop.uselist:
                        if not related:
                            continue
                        related_obj = related[0]
                    else:
                        related_obj = related
                    related_state = attributes.instance_state(related_obj)
                    if related_state in switchers:
                        uowcommit.register_object(state, False, self.passive_updates)
                        sync.populate(related_state, self.mapper, state, self.parent, self.prop.synchronize_pairs, uowcommit, self.passive_updates)

    def _pks_changed(self, uowcommit, state):
        return bool(state.key) and sync.source_modified(uowcommit, state, self.mapper, self.prop.synchronize_pairs)