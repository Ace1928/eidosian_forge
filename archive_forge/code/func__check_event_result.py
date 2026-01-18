from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def _check_event_result(self, res, model, trigger):
    if res is None:
        state_names = getattr(model, self.model_attribute)
        msg = "%sCan't trigger event '%s' from state(s) %s!" % (self.name, trigger, state_names)
        for state_name in listify(state_names):
            state = self.get_state(state_name)
            ignore = state.ignore_invalid_triggers if state.ignore_invalid_triggers is not None else self.ignore_invalid_triggers
            if not ignore:
                if self.has_trigger(trigger):
                    raise MachineError(msg)
                raise AttributeError("Do not know event named '%s'." % trigger)
        _LOGGER.warning(msg)
        res = False
    return res