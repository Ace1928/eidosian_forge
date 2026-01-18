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
def _do_send_and_ack(self, pieces, write_pipe=None, read_pipe=None):
    self._do_send(pieces, write_pipe=write_pipe)
    self._sent += 1
    msg = self._do_recv(read_pipe=read_pipe)
    su.schema_validate(msg, SCHEMAS[ACK])
    if msg != ACK:
        raise IOError('Failed receiving ack for sent message %s' % self._metrics['sent'])