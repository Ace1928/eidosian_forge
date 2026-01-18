from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def _set_state(self, state_name):
    if isinstance(state_name, list):
        return [self._set_state(value) for value in state_name]
    a_state = self.get_state(state_name)
    return a_state.value if isinstance(a_state.value, Enum) else state_name