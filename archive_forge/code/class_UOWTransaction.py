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
class UOWTransaction:
    session: Session
    transaction: SessionTransaction
    attributes: Dict[str, Any]
    deps: util.defaultdict[Mapper[Any], Set[DependencyProcessor]]
    mappers: util.defaultdict[Mapper[Any], Set[InstanceState[Any]]]

    def __init__(self, session: Session):
        self.session = session
        self.attributes = {}
        self.deps = util.defaultdict(set)
        self.mappers = util.defaultdict(set)
        self.presort_actions = {}
        self.postsort_actions = {}
        self.dependencies = set()
        self.states = {}
        self.post_update_states = util.defaultdict(lambda: (set(), set()))

    @property
    def has_work(self):
        return bool(self.states)

    def was_already_deleted(self, state):
        """Return ``True`` if the given state is expired and was deleted
        previously.
        """
        if state.expired:
            try:
                state._load_expired(state, attributes.PASSIVE_OFF)
            except orm_exc.ObjectDeletedError:
                self.session._remove_newly_deleted([state])
                return True
        return False

    def is_deleted(self, state):
        """Return ``True`` if the given state is marked as deleted
        within this uowtransaction."""
        return state in self.states and self.states[state][0]

    def memo(self, key, callable_):
        if key in self.attributes:
            return self.attributes[key]
        else:
            self.attributes[key] = ret = callable_()
            return ret

    def remove_state_actions(self, state):
        """Remove pending actions for a state from the uowtransaction."""
        isdelete = self.states[state][0]
        self.states[state] = (isdelete, True)

    def get_attribute_history(self, state, key, passive=attributes.PASSIVE_NO_INITIALIZE):
        """Facade to attributes.get_state_history(), including
        caching of results."""
        hashkey = ('history', state, key)
        if hashkey in self.attributes:
            history, state_history, cached_passive = self.attributes[hashkey]
            if not cached_passive & attributes.SQL_OK and passive & attributes.SQL_OK:
                impl = state.manager[key].impl
                history = impl.get_history(state, state.dict, attributes.PASSIVE_OFF | attributes.LOAD_AGAINST_COMMITTED | attributes.NO_RAISE)
                if history and impl.uses_objects:
                    state_history = history.as_state()
                else:
                    state_history = history
                self.attributes[hashkey] = (history, state_history, passive)
        else:
            impl = state.manager[key].impl
            history = impl.get_history(state, state.dict, passive | attributes.LOAD_AGAINST_COMMITTED | attributes.NO_RAISE)
            if history and impl.uses_objects:
                state_history = history.as_state()
            else:
                state_history = history
            self.attributes[hashkey] = (history, state_history, passive)
        return state_history

    def has_dep(self, processor):
        return (processor, True) in self.presort_actions

    def register_preprocessor(self, processor, fromparent):
        key = (processor, fromparent)
        if key not in self.presort_actions:
            self.presort_actions[key] = Preprocess(processor, fromparent)

    def register_object(self, state: InstanceState[Any], isdelete: bool=False, listonly: bool=False, cancel_delete: bool=False, operation: Optional[str]=None, prop: Optional[MapperProperty]=None) -> bool:
        if not self.session._contains_state(state):
            if not state.deleted and operation is not None:
                util.warn("Object of type %s not in session, %s operation along '%s' will not proceed" % (orm_util.state_class_str(state), operation, prop))
            return False
        if state not in self.states:
            mapper = state.manager.mapper
            if mapper not in self.mappers:
                self._per_mapper_flush_actions(mapper)
            self.mappers[mapper].add(state)
            self.states[state] = (isdelete, listonly)
        elif not listonly and (isdelete or cancel_delete):
            self.states[state] = (isdelete, False)
        return True

    def register_post_update(self, state, post_update_cols):
        mapper = state.manager.mapper.base_mapper
        states, cols = self.post_update_states[mapper]
        states.add(state)
        cols.update(post_update_cols)

    def _per_mapper_flush_actions(self, mapper):
        saves = SaveUpdateAll(self, mapper.base_mapper)
        deletes = DeleteAll(self, mapper.base_mapper)
        self.dependencies.add((saves, deletes))
        for dep in mapper._dependency_processors:
            dep.per_property_preprocessors(self)
        for prop in mapper.relationships:
            if prop.viewonly:
                continue
            dep = prop._dependency_processor
            dep.per_property_preprocessors(self)

    @util.memoized_property
    def _mapper_for_dep(self):
        """return a dynamic mapping of (Mapper, DependencyProcessor) to
        True or False, indicating if the DependencyProcessor operates
        on objects of that Mapper.

        The result is stored in the dictionary persistently once
        calculated.

        """
        return util.PopulateDict(lambda tup: tup[0]._props.get(tup[1].key) is tup[1].prop)

    def filter_states_for_dep(self, dep, states):
        """Filter the given list of InstanceStates to those relevant to the
        given DependencyProcessor.

        """
        mapper_for_dep = self._mapper_for_dep
        return [s for s in states if mapper_for_dep[s.manager.mapper, dep]]

    def states_for_mapper_hierarchy(self, mapper, isdelete, listonly):
        checktup = (isdelete, listonly)
        for mapper in mapper.base_mapper.self_and_descendants:
            for state in self.mappers[mapper]:
                if self.states[state] == checktup:
                    yield state

    def _generate_actions(self):
        """Generate the full, unsorted collection of PostSortRecs as
        well as dependency pairs for this UOWTransaction.

        """
        while True:
            ret = False
            for action in list(self.presort_actions.values()):
                if action.execute(self):
                    ret = True
            if not ret:
                break
        self.cycles = cycles = topological.find_cycles(self.dependencies, list(self.postsort_actions.values()))
        if cycles:
            convert = {rec: set(rec.per_state_flush_actions(self)) for rec in cycles}
            for edge in list(self.dependencies):
                if None in edge or edge[0].disabled or edge[1].disabled or cycles.issuperset(edge):
                    self.dependencies.remove(edge)
                elif edge[0] in cycles:
                    self.dependencies.remove(edge)
                    for dep in convert[edge[0]]:
                        self.dependencies.add((dep, edge[1]))
                elif edge[1] in cycles:
                    self.dependencies.remove(edge)
                    for dep in convert[edge[1]]:
                        self.dependencies.add((edge[0], dep))
        return {a for a in self.postsort_actions.values() if not a.disabled}.difference(cycles)

    def execute(self) -> None:
        postsort_actions = self._generate_actions()
        postsort_actions = sorted(postsort_actions, key=lambda item: item.sort_key)
        if self.cycles:
            for subset in topological.sort_as_subsets(self.dependencies, postsort_actions):
                set_ = set(subset)
                while set_:
                    n = set_.pop()
                    n.execute_aggregate(self, set_)
        else:
            for rec in topological.sort(self.dependencies, postsort_actions):
                rec.execute(self)

    def finalize_flush_changes(self) -> None:
        """Mark processed objects as clean / deleted after a successful
        flush().

        This method is called within the flush() method after the
        execute() method has succeeded and the transaction has been committed.

        """
        if not self.states:
            return
        states = set(self.states)
        isdel = {s for s, (isdelete, listonly) in self.states.items() if isdelete}
        other = states.difference(isdel)
        if isdel:
            self.session._remove_newly_deleted(isdel)
        if other:
            self.session._register_persistent(other)