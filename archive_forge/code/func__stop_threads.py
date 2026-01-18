import logging
import threading
import warnings
from debtcollector import removals
import eventlet
from eventlet import greenpool
from oslo_service import loopingcall
from oslo_utils import timeutils
def _stop_threads(self):
    self._perform_action_on_threads(lambda x: x.stop(), lambda x: LOG.exception('Error stopping thread.'))