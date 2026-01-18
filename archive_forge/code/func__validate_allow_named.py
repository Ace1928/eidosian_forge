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
def _validate_allow_named(self, val, allow_named):
    if val is None and self.allow_None:
        return
    is_hex = re.match('^#?(([0-9a-fA-F]{2}){3}|([0-9a-fA-F]){3})$', val)
    if self.allow_named:
        if not is_hex and val.lower() not in self._named_colors:
            raise ValueError(f"{_validate_error_prefix(self)} only takes RGB hex codes or named colors, received '{val}'.")
    elif not is_hex:
        raise ValueError(f'{_validate_error_prefix(self)} only accepts valid RGB hex codes, received {val!r}.')