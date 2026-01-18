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
def __compound_select__(operation, inverted=False):

    @__bind_database__
    def method(self, other):
        if inverted:
            self, other = (other, self)
        return CompoundSelectQuery(self, operation, other)
    return method