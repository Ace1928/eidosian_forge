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
class Magnitude(Number):
    """Numeric Parameter required to be in the range [0.0-1.0]."""
    _slot_defaults = _dict_update(Number._slot_defaults, default=1.0, bounds=(0.0, 1.0))

    @typing.overload
    def __init__(self, default=1.0, *, bounds=(0.0, 1.0), softbounds=None, inclusive_bounds=(True, True), step=None, set_hook=None, allow_None=False, doc=None, label=None, precedence=None, instantiate=False, constant=False, readonly=False, pickle_default_value=True, per_instance=True, allow_refs=False, nested_refs=False):
        ...

    def __init__(self, default=Undefined, *, bounds=Undefined, softbounds=Undefined, inclusive_bounds=Undefined, step=Undefined, set_hook=Undefined, **params):
        super().__init__(default=default, bounds=bounds, softbounds=softbounds, inclusive_bounds=inclusive_bounds, step=step, set_hook=set_hook, **params)