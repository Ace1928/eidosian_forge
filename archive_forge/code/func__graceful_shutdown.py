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
def _graceful_shutdown(self, *args):
    self.signal_handler.clear()
    if self.conf.graceful_shutdown_timeout and self.signal_handler.is_signal_supported('SIGALRM'):
        signal.alarm(self.conf.graceful_shutdown_timeout)
    self.stop()