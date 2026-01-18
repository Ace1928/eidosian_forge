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
class ObjectCursorWrapper(DictCursorWrapper):

    def __init__(self, cursor, constructor):
        super(ObjectCursorWrapper, self).__init__(cursor)
        self.constructor = constructor

    def process_row(self, row):
        row_dict = self._row_to_dict(row)
        return self.constructor(**row_dict)