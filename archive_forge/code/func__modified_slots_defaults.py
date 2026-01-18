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
@classmethod
def _modified_slots_defaults(cls):
    defaults = super()._modified_slots_defaults()
    defaults['objects'] = defaults.pop('_objects')
    return defaults