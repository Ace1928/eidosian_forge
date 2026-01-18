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
class Kanji(unicode_set):
    """Unicode set for Kanji Unicode Character Range"""
    _ranges = [(19968, 40895), (12288, 12351)]