import math
import warnings
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from inspect import signature
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import gamma, kv
from ..base import clone
from ..exceptions import ConvergenceWarning
from ..metrics.pairwise import pairwise_kernels
from ..utils.validation import _num_samples
class Matern(RBF):
    """Matern kernel.

    The class of Matern kernels is a generalization of the :class:`RBF`.
    It has an additional parameter :math:`\\nu` which controls the
    smoothness of the resulting function. The smaller :math:`\\nu`,
    the less smooth the approximated function is.
    As :math:`\\nu\\rightarrow\\infty`, the kernel becomes equivalent to
    the :class:`RBF` kernel. When :math:`\\nu = 1/2`, the Matérn kernel
    becomes identical to the absolute exponential kernel.
    Important intermediate values are
    :math:`\\nu=1.5` (once differentiable functions)
    and :math:`\\nu=2.5` (twice differentiable functions).

    The kernel is given by:

    .. math::
         k(x_i, x_j) =  \\frac{1}{\\Gamma(\\nu)2^{\\nu-1}}\\Bigg(
         \\frac{\\sqrt{2\\nu}}{l} d(x_i , x_j )
         \\Bigg)^\\nu K_\\nu\\Bigg(
         \\frac{\\sqrt{2\\nu}}{l} d(x_i , x_j )\\Bigg)



    where :math:`d(\\cdot,\\cdot)` is the Euclidean distance,
    :math:`K_{\\nu}(\\cdot)` is a modified Bessel function and
    :math:`\\Gamma(\\cdot)` is the gamma function.
    See [1]_, Chapter 4, Section 4.2, for details regarding the different
    variants of the Matern kernel.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    length_scale : float or ndarray of shape (n_features,), default=1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.

    nu : float, default=1.5
        The parameter nu controlling the smoothness of the learned function.
        The smaller nu, the less smooth the approximated function is.
        For nu=inf, the kernel becomes equivalent to the RBF kernel and for
        nu=0.5 to the absolute exponential kernel. Important intermediate
        values are nu=1.5 (once differentiable functions) and nu=2.5
        (twice differentiable functions). Note that values of nu not in
        [0.5, 1.5, 2.5, inf] incur a considerably higher computational cost
        (appr. 10 times higher) since they require to evaluate the modified
        Bessel function. Furthermore, in contrast to l, nu is kept fixed to
        its initial value and not optimized.

    References
    ----------
    .. [1] `Carl Edward Rasmussen, Christopher K. I. Williams (2006).
        "Gaussian Processes for Machine Learning". The MIT Press.
        <http://www.gaussianprocess.org/gpml/>`_

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import Matern
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9866...
    >>> gpc.predict_proba(X[:2,:])
    array([[0.8513..., 0.0368..., 0.1117...],
            [0.8086..., 0.0693..., 0.1220...]])
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-05, 100000.0), nu=1.5):
        super().__init__(length_scale, length_scale_bounds)
        self.nu = nu

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),                 optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X / length_scale, metric='euclidean')
        else:
            if eval_gradient:
                raise ValueError('Gradient can only be evaluated when Y is None.')
            dists = cdist(X / length_scale, Y / length_scale, metric='euclidean')
        if self.nu == 0.5:
            K = np.exp(-dists)
        elif self.nu == 1.5:
            K = dists * math.sqrt(3)
            K = (1.0 + K) * np.exp(-K)
        elif self.nu == 2.5:
            K = dists * math.sqrt(5)
            K = (1.0 + K + K ** 2 / 3.0) * np.exp(-K)
        elif self.nu == np.inf:
            K = np.exp(-dists ** 2 / 2.0)
        else:
            K = dists
            K[K == 0.0] += np.finfo(float).eps
            tmp = math.sqrt(2 * self.nu) * K
            K.fill(2 ** (1.0 - self.nu) / gamma(self.nu))
            K *= tmp ** self.nu
            K *= kv(self.nu, tmp)
        if Y is None:
            K = squareform(K)
            np.fill_diagonal(K, 1)
        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                K_gradient = np.empty((X.shape[0], X.shape[0], 0))
                return (K, K_gradient)
            if self.anisotropic:
                D = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / length_scale ** 2
            else:
                D = squareform(dists ** 2)[:, :, np.newaxis]
            if self.nu == 0.5:
                denominator = np.sqrt(D.sum(axis=2))[:, :, np.newaxis]
                divide_result = np.zeros_like(D)
                np.divide(D, denominator, out=divide_result, where=denominator != 0)
                K_gradient = K[..., np.newaxis] * divide_result
            elif self.nu == 1.5:
                K_gradient = 3 * D * np.exp(-np.sqrt(3 * D.sum(-1)))[..., np.newaxis]
            elif self.nu == 2.5:
                tmp = np.sqrt(5 * D.sum(-1))[..., np.newaxis]
                K_gradient = 5.0 / 3.0 * D * (tmp + 1) * np.exp(-tmp)
            elif self.nu == np.inf:
                K_gradient = D * K[..., np.newaxis]
            else:

                def f(theta):
                    return self.clone_with_theta(theta)(X, Y)
                return (K, _approx_fprime(self.theta, f, 1e-10))
            if not self.anisotropic:
                return (K, K_gradient[:, :].sum(-1)[:, :, np.newaxis])
            else:
                return (K, K_gradient)
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return '{0}(length_scale=[{1}], nu={2:.3g})'.format(self.__class__.__name__, ', '.join(map('{0:.3g}'.format, self.length_scale)), self.nu)
        else:
            return '{0}(length_scale={1:.3g}, nu={2:.3g})'.format(self.__class__.__name__, np.ravel(self.length_scale)[0], self.nu)