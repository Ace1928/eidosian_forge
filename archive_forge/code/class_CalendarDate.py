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
class CalendarDate(Number):
    """
    Parameter specifically allowing dates (not datetimes).
    """
    _slot_defaults = _dict_update(Number._slot_defaults, default=None)

    @typing.overload
    def __init__(self, default=None, *, bounds=None, softbounds=None, inclusive_bounds=(True, True), step=None, set_hook=None, doc=None, label=None, precedence=None, instantiate=False, constant=False, readonly=False, pickle_default_value=True, allow_None=False, per_instance=True, allow_refs=False, nested_refs=False):
        ...

    def __init__(self, default=Undefined, **kwargs):
        super().__init__(default=default, **kwargs)

    def _validate_value(self, val, allow_None):
        """
        Checks that the value is numeric and that it is within the hard
        bounds; if not, an exception is raised.
        """
        if self.allow_None and val is None:
            return
        if (not isinstance(val, dt.date) or isinstance(val, dt.datetime)) and (not (allow_None and val is None)):
            raise ValueError(f'{_validate_error_prefix(self)} only takes date types.')

    def _validate_step(self, val, step):
        if step is not None and (not isinstance(step, dt.date)):
            raise ValueError(f'{_validate_error_prefix(self, 'step')} can only be None or a date type, not {type(step)}.')

    @classmethod
    def serialize(cls, value):
        if value is None:
            return None
        return value.strftime('%Y-%m-%d')

    @classmethod
    def deserialize(cls, value):
        if value == 'null' or value is None:
            return None
        return dt.datetime.strptime(value, '%Y-%m-%d').date()