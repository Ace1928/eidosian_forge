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
class XYCoordinates(NumericTuple):
    """A NumericTuple for an X,Y coordinate."""
    _slot_defaults = _dict_update(NumericTuple._slot_defaults, default=(0.0, 0.0))

    @typing.overload
    def __init__(self, default=(0.0, 0.0), *, length=None, allow_None=False, doc=None, label=None, precedence=None, instantiate=False, constant=False, readonly=False, pickle_default_value=True, per_instance=True, allow_refs=False, nested_refs=False):
        ...

    def __init__(self, default=Undefined, **params):
        super().__init__(default=default, length=2, **params)