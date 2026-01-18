import abc
import collections
import copy
import errno
import functools
import gc
import inspect
import io
import logging
import os
import random
import signal
import sys
import time
import eventlet
from eventlet import event
from eventlet import tpool
from oslo_concurrency import lockutils
from oslo_service._i18n import _
from oslo_service import _options
from oslo_service import eventlet_backdoor
from oslo_service import systemd
from oslo_service import threadgroup
def _child_process_handle_signal(self):

    def _sigterm(*args):
        self.signal_handler.clear()
        self.launcher.stop()

    def _sighup(*args):
        self.signal_handler.clear()
        raise SignalExit(signal.SIGHUP)
    self.signal_handler.clear()
    self.signal_handler.add_handler('SIGTERM', _sigterm)
    self.signal_handler.add_handler('SIGHUP', _sighup)
    self.signal_handler.add_handler('SIGINT', self._fast_exit)