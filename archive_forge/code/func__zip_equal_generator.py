import math
import operator
import warnings
from collections import deque
from collections.abc import Sized
from functools import reduce
from itertools import (
from random import randrange, sample, choice
from sys import hexversion
def _zip_equal_generator(iterables):
    for combo in zip_longest(*iterables, fillvalue=_marker):
        for val in combo:
            if val is _marker:
                raise UnequalIterablesError()
        yield combo