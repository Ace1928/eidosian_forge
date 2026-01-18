from bisect import bisect_left
from bisect import bisect_right
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from inspect import isclass
import calendar
import collections
import datetime
import decimal
import hashlib
import itertools
import logging
import operator
import re
import socket
import struct
import sys
import threading
import time
import uuid
import warnings
def _ensure_length(self, idx):
    byte_num, byte_offset = divmod(idx, 8)
    cur_size = len(self._buffer)
    if cur_size <= byte_num:
        self._buffer.extend(b'\x00' * (byte_num + 1 - cur_size))
    return (byte_num, byte_offset)