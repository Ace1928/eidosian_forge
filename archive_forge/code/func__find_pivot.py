import ast
import collections
import contextlib
import functools
import itertools
import re
from numbers import Number
from typing import (
from more_itertools import windowed_complete
from typeguard import typechecked
from typing_extensions import Annotated, Literal
@staticmethod
def _find_pivot(words, candidates):
    pivots = (index for index in range(1, len(words) - 1) if words[index] in candidates)
    try:
        return next(pivots)
    except StopIteration:
        raise ValueError('No pivot found') from None