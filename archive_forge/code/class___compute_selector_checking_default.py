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
class __compute_selector_checking_default:

    def __call__(self, p):
        return len(p.objects) != 0

    def __repr__(self):
        return repr(self.sig)

    @property
    def sig(self):
        return None