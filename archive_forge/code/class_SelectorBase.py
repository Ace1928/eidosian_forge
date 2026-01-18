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
class SelectorBase(Parameter):
    """
    Parameter whose value must be chosen from a list of possibilities.

    Subclasses must implement get_range().
    """
    __abstract = True

    def get_range(self):
        raise NotImplementedError('get_range() must be implemented in subclasses.')