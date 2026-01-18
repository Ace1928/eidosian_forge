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
def _post_setter(self, obj, val):
    if obj is None:
        for a, v in zip(self.attribs, val):
            setattr(self.objtype, a, v)
    else:
        for a, v in zip(self.attribs, val):
            setattr(obj, a, v)