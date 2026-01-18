from concurrent import futures
import weakref
from automaton import machines
from oslo_utils import timeutils
from taskflow import logging
from taskflow import states as st
from taskflow.types import failure
from taskflow.utils import iter_utils
class MachineMemory(object):
    """State machine memory."""

    def __init__(self):
        self.next_up = set()
        self.not_done = set()
        self.failures = []
        self.done = set()

    def cancel_futures(self):
        """Attempts to cancel any not done futures."""
        for fut in self.not_done:
            fut.cancel()