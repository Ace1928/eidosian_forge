import collections
import collections.abc
import contextlib
import functools
import gzip
import itertools
import math
import operator
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import types
import weakref
import numpy as np
import matplotlib
from matplotlib import _api, _c_internal_utils
@contextlib.contextmanager
def _setattr_cm(obj, **kwargs):
    """
    Temporarily set some attributes; restore original state at context exit.
    """
    sentinel = object()
    origs = {}
    for attr in kwargs:
        orig = getattr(obj, attr, sentinel)
        if attr in obj.__dict__ or orig is sentinel:
            origs[attr] = orig
        else:
            cls_orig = getattr(type(obj), attr)
            if isinstance(cls_orig, property):
                origs[attr] = orig
            else:
                origs[attr] = sentinel
    try:
        for attr, val in kwargs.items():
            setattr(obj, attr, val)
        yield
    finally:
        for attr, orig in origs.items():
            if orig is sentinel:
                delattr(obj, attr)
            else:
                setattr(obj, attr, orig)