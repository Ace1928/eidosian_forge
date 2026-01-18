from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def _get_lin_invar_mtx(lin_invar, be, ny, names=None):
    if lin_invar is None or lin_invar == []:
        return None
    else:
        if isinstance(lin_invar[0], dict) and names:
            lin_invar = [[d[n] for n in names] for d in lin_invar]
        li_mtx = be.Matrix(lin_invar)
        if len(li_mtx.shape) != 2 or li_mtx.shape[1] != ny:
            raise ValueError('Incorrect shape of linear_invariants Matrix: %s' % str(li_mtx.shape))
        return li_mtx