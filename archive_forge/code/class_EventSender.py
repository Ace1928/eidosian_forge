import asyncore
import binascii
import collections
import errno
import functools
import hashlib
import hmac
import math
import os
import pickle
import socket
import struct
import time
import futurist
from oslo_utils import excutils
from taskflow.engines.action_engine import executor as base
from taskflow import logging
from taskflow import task as ta
from taskflow.types import notifier as nt
from taskflow.utils import iter_utils
from taskflow.utils import misc
from taskflow.utils import schema_utils as su
from taskflow.utils import threading_utils
class EventSender(object):
    """Sends event information from a child worker process to its creator."""

    def __init__(self, channel):
        self._channel = channel
        self._pid = None

    def __call__(self, event_type, details):
        if not self._channel.dead:
            if self._pid is None:
                self._pid = os.getpid()
            message = {'event_type': event_type, 'details': details, 'sent_on': time.time()}
            LOG.trace('Sending %s (from child %s)', message, self._pid)
            self._channel.send(message)