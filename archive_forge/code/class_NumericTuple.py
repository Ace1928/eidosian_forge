import collections
import copy
import datetime as dt
import glob
import inspect
import numbers
import os.path
import pathlib
import re
import sys
import typing
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from .parameterized import (
from ._utils import (
class NumericTuple(Tuple):
    """A numeric tuple Parameter (e.g. (4.5,7.6,3)) with a fixed tuple length."""

    def _validate_value(self, val, allow_None):
        super()._validate_value(val, allow_None)
        if allow_None and val is None:
            return
        for n in val:
            if _is_number(n):
                continue
            raise ValueError(f'{_validate_error_prefix(self)} only takes numeric values, not {type(n)}.')