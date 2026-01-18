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
def _produce_value(self, gen, force=False):
    """
        Return a value from gen.

        If there is no time_fn, then a new value will be returned
        (i.e. gen will be asked to produce a new value).

        If force is True, or the value of time_fn() is different from
        what it was was last time _produce_value was called, a new
        value will be produced and returned. Otherwise, the last value
        gen produced will be returned.
        """
    if hasattr(gen, '_Dynamic_time_fn'):
        time_fn = gen._Dynamic_time_fn
    else:
        time_fn = self.time_fn
    if time_fn is None or not self.time_dependent:
        value = _produce_value(gen)
        gen._Dynamic_last = value
    else:
        time = time_fn()
        if force or time != gen._Dynamic_time:
            value = _produce_value(gen)
            gen._Dynamic_last = value
            gen._Dynamic_time = time
        else:
            value = gen._Dynamic_last
    return value