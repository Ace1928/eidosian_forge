from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def _recursive_initial(self, value):
    if isinstance(value, string_types):
        path = value.split(self.state_cls.separator, 1)
        if len(path) > 1:
            state_name, suffix = path
            super(HierarchicalMachine, self.__class__).initial.fset(self, state_name)
            with self(state_name):
                self.initial = suffix
                self._initial = state_name + self.state_cls.separator + self._initial
        else:
            super(HierarchicalMachine, self.__class__).initial.fset(self, value)
    elif isinstance(value, (list, tuple)):
        return [self._recursive_initial(v) for v in value]
    else:
        super(HierarchicalMachine, self.__class__).initial.fset(self, value)
    return self._initial[0] if isinstance(self._initial, list) and len(self._initial) == 1 else self._initial