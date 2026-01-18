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
def _save_pos_integer(self, key_name, data):
    key_val = struct.unpack('!i', data)[0]
    if key_val <= 0:
        raise IOError("Invalid %s length received for key '%s', expected greater than zero length" % (key_val, key_name))
    self._memory[key_name] = key_val
    return True