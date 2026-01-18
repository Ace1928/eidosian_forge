from __future__ import division  # Many analytical derivatives depend on this
from builtins import str, next, map, zip, range, object
import math
from math import sqrt, log, isnan, isinf  # Optimization: no attribute look-up
import re
import sys
import copy
import warnings
import itertools
import inspect
import numbers
import collections
class NotParenUncert(ValueError):
    """
    Raised when a string representing an exact number or a number with
    an uncertainty indicated between parentheses was expected but not
    found.
    """