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
class _transaction(object):

    def __init__(self, db, *args, **kwargs):
        self.db = db
        self._begin_args = (args, kwargs)

    def __call__(self, fn):

        @wraps(fn)
        def inner(*args, **kwargs):
            a, k = self._begin_args
            with _transaction(self.db, *a, **k):
                return fn(*args, **kwargs)
        return inner

    def _begin(self):
        args, kwargs = self._begin_args
        self.db.begin(*args, **kwargs)

    def commit(self, begin=True):
        self.db.commit()
        if begin:
            self._begin()

    def rollback(self, begin=True):
        self.db.rollback()
        if begin:
            self._begin()

    def __enter__(self):
        if self.db.transaction_depth() == 0:
            self._begin()
        self.db.push_transaction(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        depth = self.db.transaction_depth()
        try:
            if exc_type and depth == 1:
                self.rollback(False)
            elif depth == 1:
                try:
                    self.commit(False)
                except:
                    self.rollback(False)
                    raise
        finally:
            self.db.pop_transaction()