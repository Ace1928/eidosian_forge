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
class uniform_direction_gen(multi_rv_generic):
    """A vector-valued uniform direction.

    Return a random direction (unit vector). The `dim` keyword specifies
    the dimensionality of the space.

    Methods
    -------
    rvs(dim=None, size=1, random_state=None)
        Draw random directions.

    Parameters
    ----------
    dim : scalar
        Dimension of directions.
    seed : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional

        Used for drawing random variates.
        If `seed` is `None`, the `~np.random.RandomState` singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used, seeded
        with seed.
        If `seed` is already a ``RandomState`` or ``Generator`` instance,
        then that object is used.
        Default is `None`.

    Notes
    -----
    This distribution generates unit vectors uniformly distributed on
    the surface of a hypersphere. These can be interpreted as random
    directions.
    For example, if `dim` is 3, 3D vectors from the surface of :math:`S^2`
    will be sampled.

    References
    ----------
    .. [1] Marsaglia, G. (1972). "Choosing a Point from the Surface of a
           Sphere". Annals of Mathematical Statistics. 43 (2): 645-646.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import uniform_direction
    >>> x = uniform_direction.rvs(3)
    >>> np.linalg.norm(x)
    1.

    This generates one random direction, a vector on the surface of
    :math:`S^2`.

    Alternatively, the object may be called (as a function) to return a frozen
    distribution with fixed `dim` parameter. Here,
    we create a `uniform_direction` with ``dim=3`` and draw 5 observations.
    The samples are then arranged in an array of shape 5x3.

    >>> rng = np.random.default_rng()
    >>> uniform_sphere_dist = uniform_direction(3)
    >>> unit_vectors = uniform_sphere_dist.rvs(5, random_state=rng)
    >>> unit_vectors
    array([[ 0.56688642, -0.1332634 , -0.81294566],
           [-0.427126  , -0.74779278,  0.50830044],
           [ 0.3793989 ,  0.92346629,  0.05715323],
           [ 0.36428383, -0.92449076, -0.11231259],
           [-0.27733285,  0.94410968, -0.17816678]])
    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__)

    def __call__(self, dim=None, seed=None):
        """Create a frozen n-dimensional uniform direction distribution.

        See `uniform_direction` for more information.
        """
        return uniform_direction_frozen(dim, seed=seed)

    def _process_parameters(self, dim):
        """Dimension N must be specified; it cannot be inferred."""
        if dim is None or not np.isscalar(dim) or dim < 1 or (dim != int(dim)):
            raise ValueError('Dimension of vector must be specified, and must be an integer greater than 0.')
        return int(dim)

    def rvs(self, dim, size=None, random_state=None):
        """Draw random samples from S(N-1).

        Parameters
        ----------
        dim : integer
            Dimension of space (N).
        size : int or tuple of ints, optional
            Given a shape of, for example, (m,n,k), m*n*k samples are
            generated, and packed in an m-by-n-by-k arrangement.
            Because each sample is N-dimensional, the output shape
            is (m,n,k,N). If no shape is specified, a single (N-D)
            sample is returned.
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            Pseudorandom number generator state used to generate resamples.

            If `random_state` is ``None`` (or `np.random`), the
            `numpy.random.RandomState` singleton is used.
            If `random_state` is an int, a new ``RandomState`` instance is
            used, seeded with `random_state`.
            If `random_state` is already a ``Generator`` or ``RandomState``
            instance then that instance is used.

        Returns
        -------
        rvs : ndarray
            Random direction vectors

        """
        random_state = self._get_random_state(random_state)
        if size is None:
            size = np.array([], dtype=int)
        size = np.atleast_1d(size)
        dim = self._process_parameters(dim)
        samples = _sample_uniform_direction(dim, size, random_state)
        return samples