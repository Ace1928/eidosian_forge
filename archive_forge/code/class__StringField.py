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
class _StringField(Field):

    def adapt(self, value):
        if isinstance(value, text_type):
            return value
        elif isinstance(value, bytes_type):
            return value.decode('utf-8')
        return text_type(value)

    def __add__(self, other):
        return StringExpression(self, OP.CONCAT, other)

    def __radd__(self, other):
        return StringExpression(other, OP.CONCAT, self)