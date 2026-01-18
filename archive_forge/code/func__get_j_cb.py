from __future__ import absolute_import, division, print_function
from itertools import chain
import numpy as np
from sym import Backend
from sym.util import banded_jacobian, check_transforms
from .core import NeqSys, _ensure_3args
def _get_j_cb(self):
    args = list(chain(self.x, self.params))
    kw = dict(module=self.module, dtype=object if self.module == 'mpmath' else None)
    try:
        cb = self.be.Lambdify(args, self.get_jac(), **kw)
    except TypeError:
        cb = self.be.Lambdify(args, self.get_jac())

    def j(x, params):
        return cb(np.concatenate((x, params), axis=-1))
    return j