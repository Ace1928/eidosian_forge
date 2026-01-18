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
class IPField(BigIntegerField):

    def db_value(self, val):
        if val is not None:
            return struct.unpack('!I', socket.inet_aton(val))[0]

    def python_value(self, val):
        if val is not None:
            return socket.inet_ntoa(struct.pack('!I', val))