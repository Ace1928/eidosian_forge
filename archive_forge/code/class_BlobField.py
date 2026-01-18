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
class BlobField(Field):
    field_type = 'BLOB'

    def _db_hook(self, database):
        if database is None:
            self._constructor = bytearray
        else:
            self._constructor = database.get_binary_type()

    def bind(self, model, name, set_attribute=True):
        self._constructor = bytearray
        if model._meta.database:
            if isinstance(model._meta.database, Proxy):
                model._meta.database.attach_callback(self._db_hook)
            else:
                self._db_hook(model._meta.database)
        model._meta._db_hooks.append(self._db_hook)
        return super(BlobField, self).bind(model, name, set_attribute)

    def db_value(self, value):
        if isinstance(value, text_type):
            value = value.encode('raw_unicode_escape')
        if isinstance(value, bytes_type):
            return self._constructor(value)
        return value