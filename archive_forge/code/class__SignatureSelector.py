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
class _SignatureSelector(Parameter):
    _slot_defaults = _dict_update(SelectorBase._slot_defaults, _objects=_compute_selector_default, compute_default_fn=None, check_on_set=_compute_selector_checking_default, allow_None=None, instantiate=False, default=None)

    @classmethod
    def _modified_slots_defaults(cls):
        defaults = super()._modified_slots_defaults()
        defaults['objects'] = defaults.pop('_objects')
        return defaults