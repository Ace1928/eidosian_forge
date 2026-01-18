from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def _add_dict_state(self, state, ignore_invalid_triggers, remap, **kwargs):
    if remap is not None and state['name'] in remap:
        return
    state = state.copy()
    remap = state.pop('remap', None)
    if 'ignore_invalid_triggers' not in state:
        state['ignore_invalid_triggers'] = ignore_invalid_triggers
    state_parallel = state.pop('parallel', [])
    if state_parallel:
        state_children = state_parallel
        state['initial'] = [s['name'] if isinstance(s, dict) else s for s in state_children]
    else:
        state_children = state.pop('children', state.pop('states', []))
    transitions = state.pop('transitions', [])
    new_state = self._create_state(**state)
    self.states[new_state.name] = new_state
    self._init_state(new_state)
    remapped_transitions = []
    with self(new_state.name):
        self.add_states(state_children, remap=remap, **kwargs)
        if transitions:
            self.add_transitions(transitions)
        if remap is not None:
            remapped_transitions.extend(self._remap_state(new_state, remap))
    self.add_transitions(remapped_transitions)