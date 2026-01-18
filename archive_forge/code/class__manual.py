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
class _manual(object):

    def __init__(self, db):
        self.db = db

    def __call__(self, fn):

        @wraps(fn)
        def inner(*args, **kwargs):
            with _manual(self.db):
                return fn(*args, **kwargs)
        return inner

    def __enter__(self):
        top = self.db.top_transaction()
        if top is not None and (not isinstance(top, _manual)):
            raise ValueError('Cannot enter manual commit block while a transaction is active.')
        self.db.push_transaction(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.db.pop_transaction() is not self:
            raise ValueError('Transaction stack corrupted while exiting manual commit block.')