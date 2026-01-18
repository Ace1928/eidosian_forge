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
class Chinese(unicode_set):
    """Unicode set for Chinese Unicode Character Range"""
    _ranges = [(19968, 40959), (12288, 12351)]