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
class Japanese(unicode_set):
    """Unicode set for Japanese Unicode Character Range, combining Kanji, Hiragana, and Katakana ranges"""
    _ranges = []

    class Kanji(unicode_set):
        """Unicode set for Kanji Unicode Character Range"""
        _ranges = [(19968, 40895), (12288, 12351)]

    class Hiragana(unicode_set):
        """Unicode set for Hiragana Unicode Character Range"""
        _ranges = [(12352, 12447)]

    class Katakana(unicode_set):
        """Unicode set for Katakana  Unicode Character Range"""
        _ranges = [(12448, 12543)]