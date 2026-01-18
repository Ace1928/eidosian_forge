from __future__ import annotations
import io
import sys
import typing as ty
import warnings
from functools import reduce
from operator import getitem, mul
from os.path import exists, splitext
import numpy as np
from ._compression import COMPRESSED_FILE_LIKES
from .casting import OK_FLOATS, shared_range
from .externals.oset import OrderedSet
def _is_compressed_fobj(fobj: io.IOBase) -> bool:
    """Return True if fobj represents a compressed data file-like object"""
    return isinstance(fobj, COMPRESSED_FILE_LIKES)