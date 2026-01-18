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
def _value_is_dynamic(self, obj, objtype=None):
    """
        Return True if the parameter is actually dynamic (i.e. the
        value is being generated).
        """
    return hasattr(super().__get__(obj, objtype), '_Dynamic_last')