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
def _query_val_transform(v):
    if isinstance(v, (text_type, datetime.datetime, datetime.date, datetime.time)):
        v = "'%s'" % v
    elif isinstance(v, bytes_type):
        try:
            v = v.decode('utf8')
        except UnicodeDecodeError:
            v = v.decode('raw_unicode_escape')
        v = "'%s'" % v
    elif isinstance(v, int):
        v = '%s' % int(v)
    elif v is None:
        v = 'NULL'
    else:
        v = str(v)
    return v