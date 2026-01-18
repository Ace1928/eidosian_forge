from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def _get_enum_path(self, enum_state, prefix=None):
    prefix = prefix or []
    if enum_state.name in self.states and self.states[enum_state.name].value == enum_state:
        return prefix + [enum_state.name]
    for name in self.states:
        with self(name):
            res = self._get_enum_path(enum_state, prefix=prefix + [name])
            if res:
                return res
    if not prefix:
        raise ValueError('Could not find path of {0}.'.format(enum_state))
    return None