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
class _BoundTableContext(object):

    def __init__(self, table, database):
        self.table = table
        self.database = database

    def __call__(self, fn):

        @wraps(fn)
        def inner(*args, **kwargs):
            with _BoundTableContext(self.table, self.database):
                return fn(*args, **kwargs)
        return inner

    def __enter__(self):
        self._orig_database = self.table._database
        self.table.bind(self.database)
        if self.table._model is not None:
            self.table._model.bind(self.database)
        return self.table

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.table.bind(self._orig_database)
        if self.table._model is not None:
            self.table._model.bind(self._orig_database)