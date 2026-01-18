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
@classmethod
def _get_chars_for_ranges(cls):
    ret = []
    for cc in cls.__mro__:
        if cc is unicode_set:
            break
        for rr in cc._ranges:
            ret.extend(range(rr[0], rr[-1] + 1))
    return [unichr(c) for c in sorted(set(ret))]