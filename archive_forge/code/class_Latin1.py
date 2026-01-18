import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
import pprint
import traceback
import types
from datetime import datetime
from operator import itemgetter
import itertools
from functools import wraps
from contextlib import contextmanager
class Latin1(unicode_set):
    """Unicode set for Latin-1 Unicode Character Range"""
    _ranges = [(32, 126), (160, 255)]