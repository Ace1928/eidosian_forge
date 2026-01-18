import logging
import threading
import warnings
from debtcollector import removals
import eventlet
from eventlet import greenpool
from oslo_service import loopingcall
from oslo_utils import timeutils
def _any_threads_alive(self):
    current = threading.current_thread()
    for x in self.threads[:]:
        if x.ident == current.ident:
            continue
        if not x.thread.dead:
            return True
    return False