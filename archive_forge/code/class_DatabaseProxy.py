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
class DatabaseProxy(Proxy):
    """
    Proxy implementation specifically for proxying `Database` objects.
    """
    __slots__ = ('obj', '_callbacks', '_Model')

    def connection_context(self):
        return ConnectionContext(self)

    def atomic(self, *args, **kwargs):
        return _atomic(self, *args, **kwargs)

    def manual_commit(self):
        return _manual(self)

    def transaction(self, *args, **kwargs):
        return _transaction(self, *args, **kwargs)

    def savepoint(self):
        return _savepoint(self)

    @property
    def Model(self):
        if not hasattr(self, '_Model'):

            class Meta:
                database = self
            self._Model = type('BaseModel', (Model,), {'Meta': Meta})
        return self._Model