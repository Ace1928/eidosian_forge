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
def _simple_insert(self, ctx):
    if not self._insert:
        raise self.DefaultValuesException('Error: no data to insert.')
    return self._generate_insert((self._insert,), ctx)