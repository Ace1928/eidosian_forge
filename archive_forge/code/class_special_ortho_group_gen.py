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
class special_ortho_group_gen(multi_rv_generic):
    """A Special Orthogonal matrix (SO(N)) random variable.

    Return a random rotation matrix, drawn from the Haar distribution
    (the only uniform distribution on SO(N)) with a determinant of +1.

    The `dim` keyword specifies the dimension N.

    Methods
    -------
    rvs(dim=None, size=1, random_state=None)
        Draw random samples from SO(N).

    Parameters
    ----------
    dim : scalar
        Dimension of matrices
    seed : {None, int, np.random.RandomState, np.random.Generator}, optional
        Used for drawing random variates.
        If `seed` is `None`, the `~np.random.RandomState` singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used, seeded
        with seed.
        If `seed` is already a ``RandomState`` or ``Generator`` instance,
        then that object is used.
        Default is `None`.

    Notes
    -----
    This class is wrapping the random_rot code from the MDP Toolkit,
    https://github.com/mdp-toolkit/mdp-toolkit

    Return a random rotation matrix, drawn from the Haar distribution
    (the only uniform distribution on SO(N)).
    The algorithm is described in the paper
    Stewart, G.W., "The efficient generation of random orthogonal
    matrices with an application to condition estimators", SIAM Journal
    on Numerical Analysis, 17(3), pp. 403-409, 1980.
    For more information see
    https://en.wikipedia.org/wiki/Orthogonal_matrix#Randomization

    See also the similar `ortho_group`. For a random rotation in three
    dimensions, see `scipy.spatial.transform.Rotation.random`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import special_ortho_group
    >>> x = special_ortho_group.rvs(3)

    >>> np.dot(x, x.T)
    array([[  1.00000000e+00,   1.13231364e-17,  -2.86852790e-16],
           [  1.13231364e-17,   1.00000000e+00,  -1.46845020e-16],
           [ -2.86852790e-16,  -1.46845020e-16,   1.00000000e+00]])

    >>> import scipy.linalg
    >>> scipy.linalg.det(x)
    1.0

    This generates one random matrix from SO(3). It is orthogonal and
    has a determinant of 1.

    Alternatively, the object may be called (as a function) to fix the `dim`
    parameter, returning a "frozen" special_ortho_group random variable:

    >>> rv = special_ortho_group(5)
    >>> # Frozen object with the same methods but holding the
    >>> # dimension parameter fixed.

    See Also
    --------
    ortho_group, scipy.spatial.transform.Rotation.random

    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__)

    def __call__(self, dim=None, seed=None):
        """Create a frozen SO(N) distribution.

        See `special_ortho_group_frozen` for more information.
        """
        return special_ortho_group_frozen(dim, seed=seed)

    def _process_parameters(self, dim):
        """Dimension N must be specified; it cannot be inferred."""
        if dim is None or not np.isscalar(dim) or dim <= 1 or (dim != int(dim)):
            raise ValueError('Dimension of rotation must be specified,\n                                and must be a scalar greater than 1.')
        return dim

    def rvs(self, dim, size=1, random_state=None):
        """Draw random samples from SO(N).

        Parameters
        ----------
        dim : integer
            Dimension of rotation space (N).
        size : integer, optional
            Number of samples to draw (default 1).

        Returns
        -------
        rvs : ndarray or scalar
            Random size N-dimensional matrices, dimension (size, dim, dim)

        """
        random_state = self._get_random_state(random_state)
        size = int(size)
        size = (size,) if size > 1 else ()
        dim = self._process_parameters(dim)
        H = np.empty(size + (dim, dim))
        H[..., :, :] = np.eye(dim)
        D = np.empty(size + (dim,))
        for n in range(dim - 1):
            x = random_state.normal(size=size + (dim - n,))
            xrow = x[..., None, :]
            xcol = x[..., :, None]
            norm2 = np.matmul(xrow, xcol).squeeze((-2, -1))
            x0 = x[..., 0].copy()
            D[..., n] = np.where(x0 != 0, np.sign(x0), 1)
            x[..., 0] += D[..., n] * np.sqrt(norm2)
            x /= np.sqrt((norm2 - x0 ** 2 + x[..., 0] ** 2) / 2.0)[..., None]
            H[..., :, n:] -= np.matmul(H[..., :, n:], xcol) * xrow
        D[..., -1] = (-1) ** (dim - 1) * D[..., :-1].prod(axis=-1)
        H *= D[..., :, None]
        return H