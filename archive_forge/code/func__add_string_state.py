from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def _add_string_state(self, state, on_enter, on_exit, ignore_invalid_triggers, remap, **kwargs):
    if remap is not None and state in remap:
        return
    domains = state.split(self.state_cls.separator, 1)
    if len(domains) > 1:
        try:
            self.get_state(domains[0])
        except ValueError:
            self.add_state(domains[0], on_enter=on_enter, on_exit=on_exit, ignore_invalid_triggers=ignore_invalid_triggers, **kwargs)
        with self(domains[0]):
            self.add_states(domains[1], on_enter=on_enter, on_exit=on_exit, ignore_invalid_triggers=ignore_invalid_triggers, **kwargs)
    else:
        if state in self.states:
            raise ValueError('State {0} cannot be added since it already exists.'.format(state))
        new_state = self._create_state(state, on_enter=on_enter, on_exit=on_exit, ignore_invalid_triggers=ignore_invalid_triggers, **kwargs)
        self.states[new_state.name] = new_state
        self._init_state(new_state)