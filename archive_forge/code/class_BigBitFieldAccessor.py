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
class BigBitFieldAccessor(FieldAccessor):

    def __get__(self, instance, instance_type=None):
        if instance is None:
            return self.field
        return BigBitFieldData(instance, self.name)

    def __set__(self, instance, value):
        if isinstance(value, memoryview):
            value = value.tobytes()
        elif isinstance(value, buffer_type):
            value = bytes(value)
        elif isinstance(value, bytearray):
            value = bytes_type(value)
        elif isinstance(value, BigBitFieldData):
            value = bytes_type(value._buffer)
        elif isinstance(value, text_type):
            value = value.encode('utf-8')
        elif not isinstance(value, bytes_type):
            raise ValueError('Value must be either a bytes, memoryview or BigBitFieldData instance.')
        super(BigBitFieldAccessor, self).__set__(instance, value)