from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def _enter_nested(self, root, dest, prefix_path, event_data):
    if root:
        state_name = root.pop(0)
        with event_data.machine(state_name):
            return self._enter_nested(root, dest, prefix_path, event_data)
    elif dest:
        new_states = OrderedDict()
        state_name = dest.pop(0)
        with event_data.machine(state_name):
            new_states[state_name], new_enter = self._enter_nested([], dest, prefix_path + [state_name], event_data)
            enter_partials = [partial(event_data.machine.scoped.scoped_enter, event_data, prefix_path)] + new_enter
        return (new_states, enter_partials)
    elif event_data.machine.scoped.initial:
        new_states = OrderedDict()
        enter_partials = []
        queue = []
        prefix = prefix_path
        scoped_tree = new_states
        initial_names = [i.name if hasattr(i, 'name') else i for i in listify(event_data.machine.scoped.initial)]
        initial_states = [event_data.machine.scoped.states[n] for n in initial_names]
        while True:
            event_data.scope = prefix
            for state in initial_states:
                enter_partials.append(partial(state.scoped_enter, event_data, prefix))
                scoped_tree[state.name] = OrderedDict()
                if state.initial:
                    queue.append((scoped_tree[state.name], prefix + [state.name], [state.states[i.name] if hasattr(i, 'name') else state.states[i] for i in listify(state.initial)]))
            if not queue:
                break
            scoped_tree, prefix, initial_states = queue.pop(0)
        return (new_states, enter_partials)
    else:
        return ({}, [])