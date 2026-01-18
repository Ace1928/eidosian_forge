from collections import Counter
from threading import Timer
import logging
import inspect
from ..core import MachineError, listify, State
def _process_timeout(self, event_data):
    _LOGGER.debug('%sTimeout state %s. Processing callbacks...', event_data.machine.name, self.name)
    for callback in self.on_timeout:
        event_data.machine.callback(callback, event_data)
    _LOGGER.info('%sTimeout state %s processed.', event_data.machine.name, self.name)