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
class vonmises_fisher_gen(multi_rv_generic):
    """A von Mises-Fisher variable.

    The `mu` keyword specifies the mean direction vector. The `kappa` keyword
    specifies the concentration parameter.

    Methods
    -------
    pdf(x, mu=None, kappa=1)
        Probability density function.
    logpdf(x, mu=None, kappa=1)
        Log of the probability density function.
    rvs(mu=None, kappa=1, size=1, random_state=None)
        Draw random samples from a von Mises-Fisher distribution.
    entropy(mu=None, kappa=1)
        Compute the differential entropy of the von Mises-Fisher distribution.
    fit(data)
        Fit a von Mises-Fisher distribution to data.

    Parameters
    ----------
    mu : array_like
        Mean direction of the distribution. Must be a one-dimensional unit
        vector of norm 1.
    kappa : float
        Concentration parameter. Must be positive.
    seed : {None, int, np.random.RandomState, np.random.Generator}, optional
        Used for drawing random variates.
        If `seed` is `None`, the `~np.random.RandomState` singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used, seeded
        with seed.
        If `seed` is already a ``RandomState`` or ``Generator`` instance,
        then that object is used.
        Default is `None`.

    See Also
    --------
    scipy.stats.vonmises : Von-Mises Fisher distribution in 2D on a circle
    uniform_direction : uniform distribution on the surface of a hypersphere

    Notes
    -----
    The von Mises-Fisher distribution is a directional distribution on the
    surface of the unit hypersphere. The probability density
    function of a unit vector :math:`\\mathbf{x}` is

    .. math::

        f(\\mathbf{x}) = \\frac{\\kappa^{d/2-1}}{(2\\pi)^{d/2}I_{d/2-1}(\\kappa)}
               \\exp\\left(\\kappa \\mathbf{\\mu}^T\\mathbf{x}\\right),

    where :math:`\\mathbf{\\mu}` is the mean direction, :math:`\\kappa` the
    concentration parameter, :math:`d` the dimension and :math:`I` the
    modified Bessel function of the first kind. As :math:`\\mu` represents
    a direction, it must be a unit vector or in other words, a point
    on the hypersphere: :math:`\\mathbf{\\mu}\\in S^{d-1}`. :math:`\\kappa` is a
    concentration parameter, which means that it must be positive
    (:math:`\\kappa>0`) and that the distribution becomes more narrow with
    increasing :math:`\\kappa`. In that sense, the reciprocal value
    :math:`1/\\kappa` resembles the variance parameter of the normal
    distribution.

    The von Mises-Fisher distribution often serves as an analogue of the
    normal distribution on the sphere. Intuitively, for unit vectors, a
    useful distance measure is given by the angle :math:`\\alpha` between
    them. This is exactly what the scalar product
    :math:`\\mathbf{\\mu}^T\\mathbf{x}=\\cos(\\alpha)` in the
    von Mises-Fisher probability density function describes: the angle
    between the mean direction :math:`\\mathbf{\\mu}` and the vector
    :math:`\\mathbf{x}`. The larger the angle between them, the smaller the
    probability to observe :math:`\\mathbf{x}` for this particular mean
    direction :math:`\\mathbf{\\mu}`.

    In dimensions 2 and 3, specialized algorithms are used for fast sampling
    [2]_, [3]_. For dimensions of 4 or higher the rejection sampling algorithm
    described in [4]_ is utilized. This implementation is partially based on
    the geomstats package [5]_, [6]_.

    .. versionadded:: 1.11

    References
    ----------
    .. [1] Von Mises-Fisher distribution, Wikipedia,
           https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
    .. [2] Mardia, K., and Jupp, P. Directional statistics. Wiley, 2000.
    .. [3] J. Wenzel. Numerically stable sampling of the von Mises Fisher
           distribution on S2.
           https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
    .. [4] Wood, A. Simulation of the von mises fisher distribution.
           Communications in statistics-simulation and computation 23,
           1 (1994), 157-164. https://doi.org/10.1080/03610919408813161
    .. [5] geomstats, Github. MIT License. Accessed: 06.01.2023.
           https://github.com/geomstats/geomstats
    .. [6] Miolane, N. et al. Geomstats:  A Python Package for Riemannian
           Geometry in Machine Learning. Journal of Machine Learning Research
           21 (2020). http://jmlr.org/papers/v21/19-027.html

    Examples
    --------
    **Visualization of the probability density**

    Plot the probability density in three dimensions for increasing
    concentration parameter. The density is calculated by the ``pdf``
    method.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import vonmises_fisher
    >>> from matplotlib.colors import Normalize
    >>> n_grid = 100
    >>> u = np.linspace(0, np.pi, n_grid)
    >>> v = np.linspace(0, 2 * np.pi, n_grid)
    >>> u_grid, v_grid = np.meshgrid(u, v)
    >>> vertices = np.stack([np.cos(v_grid) * np.sin(u_grid),
    ...                      np.sin(v_grid) * np.sin(u_grid),
    ...                      np.cos(u_grid)],
    ...                     axis=2)
    >>> x = np.outer(np.cos(v), np.sin(u))
    >>> y = np.outer(np.sin(v), np.sin(u))
    >>> z = np.outer(np.ones_like(u), np.cos(u))
    >>> def plot_vmf_density(ax, x, y, z, vertices, mu, kappa):
    ...     vmf = vonmises_fisher(mu, kappa)
    ...     pdf_values = vmf.pdf(vertices)
    ...     pdfnorm = Normalize(vmin=pdf_values.min(), vmax=pdf_values.max())
    ...     ax.plot_surface(x, y, z, rstride=1, cstride=1,
    ...                     facecolors=plt.cm.viridis(pdfnorm(pdf_values)),
    ...                     linewidth=0)
    ...     ax.set_aspect('equal')
    ...     ax.view_init(azim=-130, elev=0)
    ...     ax.axis('off')
    ...     ax.set_title(rf"$\\kappa={kappa}$")
    >>> fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 4),
    ...                          subplot_kw={"projection": "3d"})
    >>> left, middle, right = axes
    >>> mu = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0])
    >>> plot_vmf_density(left, x, y, z, vertices, mu, 5)
    >>> plot_vmf_density(middle, x, y, z, vertices, mu, 20)
    >>> plot_vmf_density(right, x, y, z, vertices, mu, 100)
    >>> plt.subplots_adjust(top=1, bottom=0.0, left=0.0, right=1.0, wspace=0.)
    >>> plt.show()

    As we increase the concentration parameter, the points are getting more
    clustered together around the mean direction.

    **Sampling**

    Draw 5 samples from the distribution using the ``rvs`` method resulting
    in a 5x3 array.

    >>> rng = np.random.default_rng()
    >>> mu = np.array([0, 0, 1])
    >>> samples = vonmises_fisher(mu, 20).rvs(5, random_state=rng)
    >>> samples
    array([[ 0.3884594 , -0.32482588,  0.86231516],
           [ 0.00611366, -0.09878289,  0.99509023],
           [-0.04154772, -0.01637135,  0.99900239],
           [-0.14613735,  0.12553507,  0.98126695],
           [-0.04429884, -0.23474054,  0.97104814]])

    These samples are unit vectors on the sphere :math:`S^2`. To verify,
    let us calculate their euclidean norms:

    >>> np.linalg.norm(samples, axis=1)
    array([1., 1., 1., 1., 1.])

    Plot 20 observations drawn from the von Mises-Fisher distribution for
    increasing concentration parameter :math:`\\kappa`. The red dot highlights
    the mean direction :math:`\\mu`.

    >>> def plot_vmf_samples(ax, x, y, z, mu, kappa):
    ...     vmf = vonmises_fisher(mu, kappa)
    ...     samples = vmf.rvs(20)
    ...     ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0,
    ...                     alpha=0.2)
    ...     ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c='k', s=5)
    ...     ax.scatter(mu[0], mu[1], mu[2], c='r', s=30)
    ...     ax.set_aspect('equal')
    ...     ax.view_init(azim=-130, elev=0)
    ...     ax.axis('off')
    ...     ax.set_title(rf"$\\kappa={kappa}$")
    >>> mu = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0])
    >>> fig, axes = plt.subplots(nrows=1, ncols=3,
    ...                          subplot_kw={"projection": "3d"},
    ...                          figsize=(9, 4))
    >>> left, middle, right = axes
    >>> plot_vmf_samples(left, x, y, z, mu, 5)
    >>> plot_vmf_samples(middle, x, y, z, mu, 20)
    >>> plot_vmf_samples(right, x, y, z, mu, 100)
    >>> plt.subplots_adjust(top=1, bottom=0.0, left=0.0,
    ...                     right=1.0, wspace=0.)
    >>> plt.show()

    The plots show that with increasing concentration :math:`\\kappa` the
    resulting samples are centered more closely around the mean direction.

    **Fitting the distribution parameters**

    The distribution can be fitted to data using the ``fit`` method returning
    the estimated parameters. As a toy example let's fit the distribution to
    samples drawn from a known von Mises-Fisher distribution.

    >>> mu, kappa = np.array([0, 0, 1]), 20
    >>> samples = vonmises_fisher(mu, kappa).rvs(1000, random_state=rng)
    >>> mu_fit, kappa_fit = vonmises_fisher.fit(samples)
    >>> mu_fit, kappa_fit
    (array([0.01126519, 0.01044501, 0.99988199]), 19.306398751730995)

    We see that the estimated parameters `mu_fit` and `kappa_fit` are
    very close to the ground truth parameters.

    """

    def __init__(self, seed=None):
        super().__init__(seed)

    def __call__(self, mu=None, kappa=1, seed=None):
        """Create a frozen von Mises-Fisher distribution.

        See `vonmises_fisher_frozen` for more information.
        """
        return vonmises_fisher_frozen(mu, kappa, seed=seed)

    def _process_parameters(self, mu, kappa):
        """
        Infer dimensionality from mu and ensure that mu is a one-dimensional
        unit vector and kappa positive.
        """
        mu = np.asarray(mu)
        if mu.ndim > 1:
            raise ValueError("'mu' must have one-dimensional shape.")
        if not np.allclose(np.linalg.norm(mu), 1.0):
            raise ValueError("'mu' must be a unit vector of norm 1.")
        if not mu.size > 1:
            raise ValueError("'mu' must have at least two entries.")
        kappa_error_msg = "'kappa' must be a positive scalar."
        if not np.isscalar(kappa) or kappa < 0:
            raise ValueError(kappa_error_msg)
        if float(kappa) == 0.0:
            raise ValueError("For 'kappa=0' the von Mises-Fisher distribution becomes the uniform distribution on the sphere surface. Consider using 'scipy.stats.uniform_direction' instead.")
        dim = mu.size
        return (dim, mu, kappa)

    def _check_data_vs_dist(self, x, dim):
        if x.shape[-1] != dim:
            raise ValueError("The dimensionality of the last axis of 'x' must match the dimensionality of the von Mises Fisher distribution.")
        if not np.allclose(np.linalg.norm(x, axis=-1), 1.0):
            msg = "'x' must be unit vectors of norm 1 along last dimension."
            raise ValueError(msg)

    def _log_norm_factor(self, dim, kappa):
        halfdim = 0.5 * dim
        return 0.5 * (dim - 2) * np.log(kappa) - halfdim * _LOG_2PI - np.log(ive(halfdim - 1, kappa)) - kappa

    def _logpdf(self, x, dim, mu, kappa):
        """Log of the von Mises-Fisher probability density function.

        As this function does no argument checking, it should not be
        called directly; use 'logpdf' instead.

        """
        x = np.asarray(x)
        self._check_data_vs_dist(x, dim)
        dotproducts = np.einsum('i,...i->...', mu, x)
        return self._log_norm_factor(dim, kappa) + kappa * dotproducts

    def logpdf(self, x, mu=None, kappa=1):
        """Log of the von Mises-Fisher probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability
            density function. The last axis of `x` must correspond
            to unit vectors of the same dimensionality as the distribution.
        mu : array_like, default: None
            Mean direction of the distribution. Must be a one-dimensional unit
            vector of norm 1.
        kappa : float, default: 1
            Concentration parameter. Must be positive.

        Returns
        -------
        logpdf : ndarray or scalar
            Log of the probability density function evaluated at `x`.

        """
        dim, mu, kappa = self._process_parameters(mu, kappa)
        return self._logpdf(x, dim, mu, kappa)

    def pdf(self, x, mu=None, kappa=1):
        """Von Mises-Fisher probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability
            density function. The last axis of `x` must correspond
            to unit vectors of the same dimensionality as the distribution.
        mu : array_like
            Mean direction of the distribution. Must be a one-dimensional unit
            vector of norm 1.
        kappa : float
            Concentration parameter. Must be positive.

        Returns
        -------
        pdf : ndarray or scalar
            Probability density function evaluated at `x`.

        """
        dim, mu, kappa = self._process_parameters(mu, kappa)
        return np.exp(self._logpdf(x, dim, mu, kappa))

    def _rvs_2d(self, mu, kappa, size, random_state):
        """
        In 2D, the von Mises-Fisher distribution reduces to the
        von Mises distribution which can be efficiently sampled by numpy.
        This method is much faster than the general rejection
        sampling based algorithm.

        """
        mean_angle = np.arctan2(mu[1], mu[0])
        angle_samples = random_state.vonmises(mean_angle, kappa, size=size)
        samples = np.stack([np.cos(angle_samples), np.sin(angle_samples)], axis=-1)
        return samples

    def _rvs_3d(self, kappa, size, random_state):
        """
        Generate samples from a von Mises-Fisher distribution
        with mu = [1, 0, 0] and kappa. Samples then have to be
        rotated towards the desired mean direction mu.
        This method is much faster than the general rejection
        sampling based algorithm.
        Reference: https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf

        """
        if size is None:
            sample_size = 1
        else:
            sample_size = size
        x = random_state.random(sample_size)
        x = 1.0 + np.log(x + (1.0 - x) * np.exp(-2 * kappa)) / kappa
        temp = np.sqrt(1.0 - np.square(x))
        uniformcircle = _sample_uniform_direction(2, sample_size, random_state)
        samples = np.stack([x, temp * uniformcircle[..., 0], temp * uniformcircle[..., 1]], axis=-1)
        if size is None:
            samples = np.squeeze(samples)
        return samples

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

    def _rvs(self, dim, mu, kappa, size, random_state):
        if dim == 2:
            samples = self._rvs_2d(mu, kappa, size, random_state)
        elif dim == 3:
            samples = self._rvs_3d(kappa, size, random_state)
        else:
            samples = self._rejection_sampling(dim, kappa, size, random_state)
        if dim != 2:
            samples = self._rotate_samples(samples, mu, dim)
        return samples

    def rvs(self, mu=None, kappa=1, size=1, random_state=None):
        """Draw random samples from a von Mises-Fisher distribution.

        Parameters
        ----------
        mu : array_like
            Mean direction of the distribution. Must be a one-dimensional unit
            vector of norm 1.
        kappa : float
            Concentration parameter. Must be positive.
        size : int or tuple of ints, optional
            Given a shape of, for example, (m,n,k), m*n*k samples are
            generated, and packed in an m-by-n-by-k arrangement.
            Because each sample is N-dimensional, the output shape
            is (m,n,k,N). If no shape is specified, a single (N-D)
            sample is returned.
        random_state : {None, int, np.random.RandomState, np.random.Generator},
                        optional
            Used for drawing random variates.
            If `seed` is `None`, the `~np.random.RandomState` singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used, seeded
            with seed.
            If `seed` is already a ``RandomState`` or ``Generator`` instance,
            then that object is used.
            Default is `None`.

        Returns
        -------
        rvs : ndarray
            Random variates of shape (`size`, `N`), where `N` is the
            dimension of the distribution.

        """
        dim, mu, kappa = self._process_parameters(mu, kappa)
        random_state = self._get_random_state(random_state)
        samples = self._rvs(dim, mu, kappa, size, random_state)
        return samples

    def _entropy(self, dim, kappa):
        halfdim = 0.5 * dim
        return -self._log_norm_factor(dim, kappa) - kappa * ive(halfdim, kappa) / ive(halfdim - 1, kappa)

    def entropy(self, mu=None, kappa=1):
        """Compute the differential entropy of the von Mises-Fisher
        distribution.

        Parameters
        ----------
        mu : array_like, default: None
            Mean direction of the distribution. Must be a one-dimensional unit
            vector of norm 1.
        kappa : float, default: 1
            Concentration parameter. Must be positive.

        Returns
        -------
        h : scalar
            Entropy of the von Mises-Fisher distribution.

        """
        dim, _, kappa = self._process_parameters(mu, kappa)
        return self._entropy(dim, kappa)

    def fit(self, x):
        """Fit the von Mises-Fisher distribution to data.

        Parameters
        ----------
        x : array-like
            Data the distribution is fitted to. Must be two dimensional.
            The second axis of `x` must be unit vectors of norm 1 and
            determine the dimensionality of the fitted
            von Mises-Fisher distribution.

        Returns
        -------
        mu : ndarray
            Estimated mean direction.
        kappa : float
            Estimated concentration parameter.

        """
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError("'x' must be two dimensional.")
        if not np.allclose(np.linalg.norm(x, axis=-1), 1.0):
            msg = "'x' must be unit vectors of norm 1 along last dimension."
            raise ValueError(msg)
        dim = x.shape[-1]
        dirstats = directional_stats(x)
        mu = dirstats.mean_direction
        r = dirstats.mean_resultant_length
        halfdim = 0.5 * dim

        def solve_for_kappa(kappa):
            bessel_vals = ive([halfdim, halfdim - 1], kappa)
            return bessel_vals[0] / bessel_vals[1] - r
        root_res = root_scalar(solve_for_kappa, method='brentq', bracket=(1e-08, 1000000000.0))
        kappa = root_res.root
        return (mu, kappa)