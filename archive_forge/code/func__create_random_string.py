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
def _create_random_string(desired_length):
    if desired_length <= 0:
        return b''
    data_length = int(math.ceil(desired_length / 2.0))
    data = os.urandom(data_length)
    hex_data = binascii.hexlify(data)
    return hex_data[0:desired_length]