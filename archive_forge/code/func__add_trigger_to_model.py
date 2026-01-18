from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def _add_trigger_to_model(self, trigger, model):
    trig_func = partial(self.trigger_event, model, trigger)
    self._add_may_transition_func_for_trigger(trigger, model)
    if trigger.startswith('to_') and self.state_cls.separator != '_':
        path = trigger[3:].split(self.state_cls.separator)
        if hasattr(model, 'to_' + path[0]):
            getattr(model, 'to_' + path[0]).add(trig_func, path[1:])
        else:
            self._checked_assignment(model, 'to_' + path[0], FunctionWrapper(trig_func, path[1:]))
    else:
        self._checked_assignment(model, trigger, trig_func)