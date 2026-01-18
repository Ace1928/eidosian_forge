from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
@staticmethod
def _kwargs_roots_from_roots_cb(roots_cb, kwargs, x, _y, _p, be):
    if roots_cb is not None:
        if 'roots' in kwargs:
            raise ValueError('Keyword argument ``roots`` already given.')
        try:
            roots = roots_cb(x, _y, _p, be)
        except TypeError:
            roots = _ensure_4args(roots_cb)(x, _y, _p, be)
        kwargs['roots'] = roots