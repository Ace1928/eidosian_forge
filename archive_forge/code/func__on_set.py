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
def _on_set(self, attribute, old, new):
    super()._on_set(attribute, new, old)
    if attribute == 'path':
        self.update(path=new)