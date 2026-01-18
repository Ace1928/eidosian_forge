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
def __join__(join_type=JOIN.INNER, inverted=False):

    def method(self, other):
        if inverted:
            self, other = (other, self)
        return Join(self, other, join_type=join_type)
    return method