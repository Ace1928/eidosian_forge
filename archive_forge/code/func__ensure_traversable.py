import collections
import contextlib
import itertools
import pathlib
import operator
import re
import warnings
from . import abc
from ._itertools import only
from .compat.py39 import ZipPath
def _ensure_traversable(path):
    """
    Convert deprecated string arguments to traversables (pathlib.Path).

    Remove with Python 3.15.
    """
    if not isinstance(path, str):
        return path
    warnings.warn('String arguments are deprecated. Pass a Traversable instead.', DeprecationWarning, stacklevel=3)
    return pathlib.Path(path)