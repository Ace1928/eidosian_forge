import logging
import threading
import warnings
from debtcollector import removals
import eventlet
from eventlet import greenpool
from oslo_service import loopingcall
from oslo_utils import timeutils
def _on_thread_done(_greenthread, group, thread):
    """Callback function to be passed to GreenThread.link() when we spawn().

    Calls the :class:`ThreadGroup` to notify it to remove this thread from
    the associated group.
    """
    group.thread_done(thread)