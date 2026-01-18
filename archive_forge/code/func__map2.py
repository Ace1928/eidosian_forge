from __future__ import absolute_import, division, print_function
from itertools import chain
import numpy as np
from sym import Backend
from sym.util import banded_jacobian, check_transforms
from .core import NeqSys, _ensure_3args
def _map2(cb, iterable):
    if cb is None:
        return iterable
    else:
        return map(cb, iterable)