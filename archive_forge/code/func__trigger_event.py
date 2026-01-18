from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def _trigger_event(self, event_data, trigger):
    try:
        with self():
            res = self._trigger_event_nested(event_data, trigger, None)
        event_data.result = self._check_event_result(res, event_data.model, trigger)
    except Exception as err:
        event_data.error = err
        if self.on_exception:
            self.callbacks(self.on_exception, event_data)
        else:
            raise
    finally:
        try:
            self.callbacks(self.finalize_event, event_data)
            _LOGGER.debug('%sExecuted machine finalize callbacks', self.name)
        except Exception as err:
            _LOGGER.error('%sWhile executing finalize callbacks a %s occurred: %s.', self.name, type(err).__name__, str(err))
    return event_data.result