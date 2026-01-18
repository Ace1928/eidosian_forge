from __future__ import absolute_import
from functools import partial
import numpy.linalg as npla
from .numpy_wrapper import wrap_namespace
from . import numpy_wrapper as anp
from autograd.extend import defvjp, defjvp
def _matrix_diag(a):
    reps = anp.array(a.shape)
    reps[:-1] = 1
    reps[-1] = a.shape[-1]
    newshape = list(a.shape) + [a.shape[-1]]
    return _diag(anp.tile(a, reps).reshape(newshape))