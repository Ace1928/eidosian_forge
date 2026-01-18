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
class ModelNamedTupleCursorWrapper(ModelTupleCursorWrapper):

    def initialize(self):
        self._initialize_columns()
        attributes = []
        for i in range(self.ncols):
            attributes.append(self.columns[i])
        self.tuple_class = collections.namedtuple('Row', attributes)
        self.constructor = lambda row: self.tuple_class(*row)