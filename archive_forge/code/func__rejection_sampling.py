import math
import numpy as np
import scipy.linalg
from scipy._lib import doccer
from scipy.special import (gammaln, psi, multigammaln, xlogy, entr, betaln,
from scipy._lib._util import check_random_state, _lazywhere
from scipy.linalg.blas import drot, get_blas_funcs
from ._continuous_distns import norm
from ._discrete_distns import binom
from . import _mvn, _covariance, _rcont
from ._qmvnt import _qmvt
from ._morestats import directional_stats
from scipy.optimize import root_scalar
def _rejection_sampling(self, dim, kappa, size, random_state):
    """
        Generate samples from a n-dimensional von Mises-Fisher distribution
        with mu = [1, 0, ..., 0] and kappa via rejection sampling.
        Samples then have to be rotated towards the desired mean direction mu.
        Reference: https://doi.org/10.1080/03610919408813161
        """
    dim_minus_one = dim - 1
    if size is not None:
        if not np.iterable(size):
            size = (size,)
        n_samples = math.prod(size)
    else:
        n_samples = 1
    sqrt = np.sqrt(4 * kappa ** 2.0 + dim_minus_one ** 2)
    envelop_param = (-2 * kappa + sqrt) / dim_minus_one
    if envelop_param == 0:
        envelop_param = dim_minus_one / 4 * kappa ** (-1.0) - dim_minus_one ** 3 / 64 * kappa ** (-3.0)
    node = (1.0 - envelop_param) / (1.0 + envelop_param)
    correction = kappa * node + dim_minus_one * (np.log(4) + np.log(envelop_param) - 2 * np.log1p(envelop_param))
    n_accepted = 0
    x = np.zeros((n_samples,))
    halfdim = 0.5 * dim_minus_one
    while n_accepted < n_samples:
        sym_beta = random_state.beta(halfdim, halfdim, size=n_samples - n_accepted)
        coord_x = (1 - (1 + envelop_param) * sym_beta) / (1 - (1 - envelop_param) * sym_beta)
        accept_tol = random_state.random(n_samples - n_accepted)
        criterion = kappa * coord_x + dim_minus_one * np.log((1 + envelop_param - coord_x + coord_x * envelop_param) / (1 + envelop_param)) - correction > np.log(accept_tol)
        accepted_iter = np.sum(criterion)
        x[n_accepted:n_accepted + accepted_iter] = coord_x[criterion]
        n_accepted += accepted_iter
    coord_rest = _sample_uniform_direction(dim_minus_one, n_accepted, random_state)
    coord_rest = np.einsum('...,...i->...i', np.sqrt(1 - x ** 2), coord_rest)
    samples = np.concatenate([x[..., None], coord_rest], axis=1)
    if size is not None:
        samples = samples.reshape(size + (dim,))
    else:
        samples = np.squeeze(samples)
    return samples