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
def _ensure_value_is_in_objects(self, val):
    """
        Make sure that the provided value is present on the objects list.
        Subclasses can override if they support multiple items on a list,
        to check each item instead.
        """
    if not val in self.objects:
        self._objects.append(val)