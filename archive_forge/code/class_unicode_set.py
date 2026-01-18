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
class unicode_set(object):
    """
    A set of Unicode characters, for language-specific strings for
    ``alphas``, ``nums``, ``alphanums``, and ``printables``.
    A unicode_set is defined by a list of ranges in the Unicode character
    set, in a class attribute ``_ranges``, such as::

        _ranges = [(0x0020, 0x007e), (0x00a0, 0x00ff),]

    A unicode set can also be defined using multiple inheritance of other unicode sets::

        class CJK(Chinese, Japanese, Korean):
            pass
    """
    _ranges = []

    @classmethod
    def _get_chars_for_ranges(cls):
        ret = []
        for cc in cls.__mro__:
            if cc is unicode_set:
                break
            for rr in cc._ranges:
                ret.extend(range(rr[0], rr[-1] + 1))
        return [unichr(c) for c in sorted(set(ret))]

    @_lazyclassproperty
    def printables(cls):
        """all non-whitespace characters in this range"""
        return u''.join(filterfalse(unicode.isspace, cls._get_chars_for_ranges()))

    @_lazyclassproperty
    def alphas(cls):
        """all alphabetic characters in this range"""
        return u''.join(filter(unicode.isalpha, cls._get_chars_for_ranges()))

    @_lazyclassproperty
    def nums(cls):
        """all numeric digit characters in this range"""
        return u''.join(filter(unicode.isdigit, cls._get_chars_for_ranges()))

    @_lazyclassproperty
    def alphanums(cls):
        """all alphanumeric characters in this range"""
        return cls.alphas + cls.nums