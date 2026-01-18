import re
import warnings
from numpy import asarray, array, zeros, isscalar, real, imag, vstack
from . import _vode
from . import _dop
from . import _lsoda
class IntegratorConcurrencyError(RuntimeError):
    """
    Failure due to concurrent usage of an integrator that can be used
    only for a single problem at a time.

    """

    def __init__(self, name):
        msg = 'Integrator `%s` can be used to solve only a single problem at a time. If you want to integrate multiple problems, consider using a different integrator (see `ode.set_integrator`)' % name
        RuntimeError.__init__(self, msg)