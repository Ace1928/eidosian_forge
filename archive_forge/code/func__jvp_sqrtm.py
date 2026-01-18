from __future__ import division
import scipy.linalg
import autograd.numpy as anp
from autograd.numpy.numpy_wrapper import wrap_namespace
from autograd.extend import defvjp, defvjp_argnums, defjvp, defjvp_argnums
def _jvp_sqrtm(dA, ans, A, disp=True, blocksize=64):
    assert disp, 'sqrtm jvp not implemented for disp=False'
    return solve_sylvester(ans, ans, dA)