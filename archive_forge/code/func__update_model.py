from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
@staticmethod
def _update_model(event_data, tree):
    model_states = _build_state_list(tree, event_data.machine.state_cls.separator)
    with event_data.machine():
        event_data.machine.set_state(model_states, event_data.model)
        states = event_data.machine.get_states(listify(model_states))
        event_data.state = states[0] if len(states) == 1 else states