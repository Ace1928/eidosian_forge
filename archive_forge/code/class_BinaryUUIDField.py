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
class BinaryUUIDField(BlobField):
    field_type = 'UUIDB'

    def db_value(self, value):
        if isinstance(value, bytes) and len(value) == 16:
            return self._constructor(value)
        elif isinstance(value, basestring) and len(value) == 32:
            value = uuid.UUID(hex=value)
        if isinstance(value, uuid.UUID):
            return self._constructor(value.bytes)
        elif value is not None:
            raise ValueError('value for binary UUID field must be UUID(), a hexadecimal string, or a bytes object.')

    def python_value(self, value):
        if isinstance(value, uuid.UUID):
            return value
        elif isinstance(value, memoryview):
            value = value.tobytes()
        elif value and (not isinstance(value, bytes)):
            value = bytes(value)
        return uuid.UUID(bytes=value) if value is not None else None