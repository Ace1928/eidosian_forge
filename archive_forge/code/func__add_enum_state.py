from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def _add_enum_state(self, state, on_enter, on_exit, ignore_invalid_triggers, remap, **kwargs):
    if remap is not None and state.name in remap:
        return
    if self.state_cls.separator in state.name:
        raise ValueError("State '{0}' contains '{1}' which is used as state name separator. Consider changing the NestedState.separator to avoid this issue.".format(state.name, self.state_cls.separator))
    if state.name in self.states:
        raise ValueError('State {0} cannot be added since it already exists.'.format(state.name))
    new_state = self._create_state(state, on_enter=on_enter, on_exit=on_exit, ignore_invalid_triggers=ignore_invalid_triggers, **kwargs)
    self.states[new_state.name] = new_state
    self._init_state(new_state)