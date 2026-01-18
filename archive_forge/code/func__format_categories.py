from __future__ import print_function, division, absolute_import
import ctypes
import operator
from collections import OrderedDict
from math import ceil
from datashader import datashape
import numpy as np
from .internal_utils import IndexCallable, isidentifier
def _format_categories(cats, n=10):
    return '[%s%s]' % (', '.join(map(repr, cats[:n])), ', ...' if len(cats) > n else '')