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
def _rotate_samples(self, samples, mu, dim):
    """A QR decomposition is used to find the rotation that maps the
        north pole (1, 0,...,0) to the vector mu. This rotation is then
        applied to all samples.

        Parameters
        ----------
        samples: array_like, shape = [..., n]
        mu : array-like, shape=[n, ]
            Point to parametrise the rotation.

        Returns
        -------
        samples : rotated samples

        """
    base_point = np.zeros((dim,))
    base_point[0] = 1.0
    embedded = np.concatenate([mu[None, :], np.zeros((dim - 1, dim))])
    rotmatrix, _ = np.linalg.qr(np.transpose(embedded))
    if np.allclose(np.matmul(rotmatrix, base_point[:, None])[:, 0], mu):
        rotsign = 1
    else:
        rotsign = -1
    samples = np.einsum('ij,...j->...i', rotmatrix, samples) * rotsign
    return samples