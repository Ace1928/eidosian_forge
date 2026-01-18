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
class WrappedNode(ColumnBase):

    def __init__(self, node):
        self.node = node
        self._coerce = getattr(node, '_coerce', True)
        self._converter = getattr(node, '_converter', None)

    def is_alias(self):
        return self.node.is_alias()

    def unwrap(self):
        return self.node.unwrap()