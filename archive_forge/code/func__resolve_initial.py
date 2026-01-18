from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def _resolve_initial(self, models, state_name_path, prefix=None):
    prefix = prefix or []
    if state_name_path:
        state_name = state_name_path.pop(0)
        with self(state_name):
            return self._resolve_initial(models, state_name_path, prefix=prefix + [state_name])
    if self.scoped.initial:
        entered_states = []
        for initial_state_name in listify(self.scoped.initial):
            with self(initial_state_name):
                entered_states.append(self._resolve_initial(models, [], prefix=prefix + [self.scoped.name]))
        return entered_states if len(entered_states) > 1 else entered_states[0]
    return self.state_cls.separator.join(prefix)