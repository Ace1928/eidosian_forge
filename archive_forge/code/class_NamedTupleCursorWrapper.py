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
class NamedTupleCursorWrapper(CursorWrapper):

    def initialize(self):
        description = self.cursor.description
        self.tuple_class = collections.namedtuple('Row', [t[0][t[0].rfind('.') + 1:].strip('()"`') for t in description])

    def process_row(self, row):
        return self.tuple_class(*row)