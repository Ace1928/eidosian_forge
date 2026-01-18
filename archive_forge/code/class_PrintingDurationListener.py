import itertools
import time
from oslo_utils import timeutils
from taskflow.engines.action_engine import compiler as co
from taskflow import exceptions as exc
from taskflow.listeners import base
from taskflow import logging
from taskflow import states
class PrintingDurationListener(DurationListener):
    """Listener that prints the duration as well as recording it."""

    def __init__(self, engine, printer=None):
        super(PrintingDurationListener, self).__init__(engine)
        if printer is None:
            self._printer = _printer
        else:
            self._printer = printer

    def _record_ending(self, timer, item_type, item_name, state):
        super(PrintingDurationListener, self)._record_ending(timer, item_type, item_name, state)
        self._printer("It took %s '%s' %0.2f seconds to finish." % (item_type, item_name, timer.elapsed()))

    def _receiver(self, item_type, item_name, state):
        super(PrintingDurationListener, self)._receiver(item_type, item_name, state)
        if state in STARTING_STATES:
            self._printer("'%s' %s started." % (item_name, item_type))