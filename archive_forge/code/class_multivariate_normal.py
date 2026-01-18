from collections.abc import Sequence
import numpy as np
import xarray as xr
from numpy.linalg import LinAlgError
from scipy import special, stats
from . import _remove_indexes_to_reduce
from .linalg import cholesky, eigh
class multivariate_normal:
    """An xarray aware multivariate normal random variable.

    Notes
    -----
    This currently is **not** a wrapper of :class:`scipy.stats.multivariate_normal`.
    It only implements a subset of the features. The reason for reimplementing
    some of the features instead of wrapping scipy or numpy is that neither
    is capable of handling batched inputs yet.
    """

    def __init__(self, mean=None, cov=None, dims=None):
        """Initialize the multivariate_normal class."""
        self.mean = mean
        self.cov = cov
        self.dims = dims

    def _process_inputs(self, mean, cov, dims):
        base_error = 'No value found for parameter {param}. It needs to be defined either at class initialization time or when calling its methods'
        mean = mean if mean is not None else self.mean
        if mean is None:
            raise ValueError(base_error.format(param='mean'))
        cov = cov if cov is not None else self.cov
        if cov is None:
            raise ValueError(base_error.format(param='cov'))
        dims = dims if dims is not None else self.dims
        if dims is None:
            raise ValueError(base_error.format(param='dims'))
        if len(dims) != 2:
            raise ValueError('dims must be an iterable of length 2')
        dim1, dim2 = dims
        if dim1 not in mean.dims:
            raise ValueError(f'{dim1} not found in DataArray provided as mean')
        if dim1 not in cov.dims or dim2 not in cov.dims:
            raise ValueError('Some dimensions provided in `dims` were not found in DataArray provided as `mean`')
        return (mean, cov, dims)

    def rvs(self, mean=None, cov=None, dims=None, *, size=1, rv_dims=None, random_state=None):
        """Generate random samples from a multivariate normal."""
        mean, cov, dims = self._process_inputs(mean, cov, dims)
        dim1, dim2 = dims
        try:
            cov_chol = cholesky(cov, dims=dims)
        except LinAlgError:
            k = len(cov[dim1])
            eye = xr.DataArray(np.eye(k), dims=list(dims))
            cov_chol = cholesky(cov + 1e-10 * eye, dims=dims)
        std_norm = XrContinuousRV(stats.norm, xr.zeros_like(mean.rename({dim1: dim2})), 1)
        samples = std_norm.rvs(size=size, dims=rv_dims, random_state=random_state)
        return mean + xr.dot(cov_chol, samples, dims=dim2)

    def logpdf(self, x, mean=None, cov=None, dims=None):
        """Evaluate the logarithm of the multivariate normal probability density function."""
        x = _asdataarray(x, 'point')
        mean, cov, dims = self._process_inputs(mean, cov, dims)
        dim1, dim2 = dims
        k = len(mean[dim1])
        vals, vecs = eigh(cov, dims=dims)
        logdet_cov = np.log(vals).sum(dim=dim2)
        u_mat = vecs * np.sqrt(1.0 / vals.rename({dim2: dim1}))
        x_mu = x - mean
        maha = np.square(xr.dot(x_mu.rename({dim1: dim2}), u_mat, dims=dim2)).sum(dim=dim1)
        return -0.5 * (k * np.log(2 * np.pi) + maha + logdet_cov)

    def pdf(self, x, mean=None, cov=None, dims=None):
        """Evaluate the multivariate normal probability density function."""
        return np.exp(self.logpdf(x, mean, cov, dims))