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
def _drop_indexes(self, safe=True):
    return [self._drop_index(index, safe) for index in self.model._meta.fields_to_index() if isinstance(index, Index)]