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
class _BoundModelsContext(object):

    def __init__(self, models, database, bind_refs, bind_backrefs):
        self.models = models
        self.database = database
        self.bind_refs = bind_refs
        self.bind_backrefs = bind_backrefs

    def __enter__(self):
        self._orig_database = []
        for model in self.models:
            self._orig_database.append(model._meta.database)
            model.bind(self.database, self.bind_refs, self.bind_backrefs, _exclude=set(self.models))
        return self.models

    def __exit__(self, exc_type, exc_val, exc_tb):
        for model, db in zip(self.models, self._orig_database):
            model.bind(db, self.bind_refs, self.bind_backrefs, _exclude=set(self.models))