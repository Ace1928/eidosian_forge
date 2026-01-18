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
class HookList(List):
    """
    Parameter whose value is a list of callable objects.

    This type of List Parameter is typically used to provide a place
    for users to register a set of commands to be called at a
    specified place in some sequence of processing steps.
    """
    __slots__ = ['class_', 'bounds']

    def _validate_value(self, val, allow_None):
        super()._validate_value(val, allow_None)
        if allow_None and val is None:
            return
        for v in val:
            if callable(v):
                continue
            raise ValueError(f'{_validate_error_prefix(self)} items must be callable, not {v!r}.')