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
def _save_and_validate_magic(self, data):
    magic_header = struct.unpack('!i', data)[0]
    if magic_header != MAGIC_HEADER:
        raise IOError('Invalid magic header received, expected 0x%x but got 0x%x for message %s' % (MAGIC_HEADER, magic_header, self.msg_count + 1))
    self._memory['magic'] = magic_header
    return True