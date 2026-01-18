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
class ModelObjectCursorWrapper(ModelDictCursorWrapper):

    def __init__(self, cursor, model, select, constructor):
        self.constructor = constructor
        self.is_model = is_model(constructor)
        super(ModelObjectCursorWrapper, self).__init__(cursor, model, select)

    def process_row(self, row):
        data = super(ModelObjectCursorWrapper, self).process_row(row)
        if self.is_model:
            obj = self.constructor(__no_default__=1, **data)
            obj._dirty.clear()
            return obj
        else:
            return self.constructor(**data)