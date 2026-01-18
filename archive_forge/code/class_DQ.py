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
class DQ(ColumnBase):

    def __init__(self, **query):
        super(DQ, self).__init__()
        self.query = query
        self._negated = False

    @Node.copy
    def __invert__(self):
        self._negated = not self._negated

    def clone(self):
        node = DQ(**self.query)
        node._negated = self._negated
        return node