import operator
import math
from .backend import MPZ_ZERO, MPZ_ONE, BACKEND, xrange, exec_
from .libintmath import gcd
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
from .libintmath import ifac
from .gammazeta import mpf_gamma_int, mpf_euler, euler_fixed
def _hypsum(coeffs, z, prec, wp, epsshift, magnitude_check, **kwargs):
    return hypsum_internal(p, q, param_types, ztype, coeffs, z, prec, wp, epsshift, magnitude_check, kwargs)