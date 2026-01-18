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
def __pragma__(name):

    def __get__(self):
        return self.pragma(name)

    def __set__(self, value):
        return self.pragma(name, value)
    return property(__get__, __set__)