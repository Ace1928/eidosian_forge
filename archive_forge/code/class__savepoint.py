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
class _savepoint(object):

    def __init__(self, db, sid=None):
        self.db = db
        self.sid = sid or 's' + uuid.uuid4().hex
        self.quoted_sid = self.sid.join(self.db.quote)

    def __call__(self, fn):

        @wraps(fn)
        def inner(*args, **kwargs):
            with _savepoint(self.db):
                return fn(*args, **kwargs)
        return inner

    def _begin(self):
        self.db.execute_sql('SAVEPOINT %s;' % self.quoted_sid)

    def commit(self, begin=True):
        self.db.execute_sql('RELEASE SAVEPOINT %s;' % self.quoted_sid)
        if begin:
            self._begin()

    def rollback(self):
        self.db.execute_sql('ROLLBACK TO SAVEPOINT %s;' % self.quoted_sid)

    def __enter__(self):
        self._begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.rollback()
        else:
            try:
                self.commit(begin=False)
            except:
                self.rollback()
                raise