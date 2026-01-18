import math
import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _standard_normal, lazy_property
class MultivariateNormal(Distribution):
    """
    Creates a multivariate normal (also called Gaussian) distribution
    parameterized by a mean vector and a covariance matrix.

    The multivariate normal distribution can be parameterized either
    in terms of a positive definite covariance matrix :math:`\\mathbf{\\Sigma}`
    or a positive definite precision matrix :math:`\\mathbf{\\Sigma}^{-1}`
    or a lower-triangular matrix :math:`\\mathbf{L}` with positive-valued
    diagonal entries, such that
    :math:`\\mathbf{\\Sigma} = \\mathbf{L}\\mathbf{L}^\\top`. This triangular matrix
    can be obtained via e.g. Cholesky decomposition of the covariance.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = MultivariateNormal(torch.zeros(2), torch.eye(2))
        >>> m.sample()  # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
        tensor([-0.2102, -0.5429])

    Args:
        loc (Tensor): mean of the distribution
        covariance_matrix (Tensor): positive-definite covariance matrix
        precision_matrix (Tensor): positive-definite precision matrix
        scale_tril (Tensor): lower-triangular factor of covariance, with positive-valued diagonal

    Note:
        Only one of :attr:`covariance_matrix` or :attr:`precision_matrix` or
        :attr:`scale_tril` can be specified.

        Using :attr:`scale_tril` will be more efficient: all computations internally
        are based on :attr:`scale_tril`. If :attr:`covariance_matrix` or
        :attr:`precision_matrix` is passed instead, it is only used to compute
        the corresponding lower triangular matrices using a Cholesky decomposition.
    """
    arg_constraints = {'loc': constraints.real_vector, 'covariance_matrix': constraints.positive_definite, 'precision_matrix': constraints.positive_definite, 'scale_tril': constraints.lower_cholesky}
    support = constraints.real_vector
    has_rsample = True

    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None):
        if loc.dim() < 1:
            raise ValueError('loc must be at least one-dimensional.')
        if (covariance_matrix is not None) + (scale_tril is not None) + (precision_matrix is not None) != 1:
            raise ValueError('Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified.')
        if scale_tril is not None:
            if scale_tril.dim() < 2:
                raise ValueError('scale_tril matrix must be at least two-dimensional, with optional leading batch dimensions')
            batch_shape = torch.broadcast_shapes(scale_tril.shape[:-2], loc.shape[:-1])
            self.scale_tril = scale_tril.expand(batch_shape + (-1, -1))
        elif covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError('covariance_matrix must be at least two-dimensional, with optional leading batch dimensions')
            batch_shape = torch.broadcast_shapes(covariance_matrix.shape[:-2], loc.shape[:-1])
            self.covariance_matrix = covariance_matrix.expand(batch_shape + (-1, -1))
        else:
            if precision_matrix.dim() < 2:
                raise ValueError('precision_matrix must be at least two-dimensional, with optional leading batch dimensions')
            batch_shape = torch.broadcast_shapes(precision_matrix.shape[:-2], loc.shape[:-1])
            self.precision_matrix = precision_matrix.expand(batch_shape + (-1, -1))
        self.loc = loc.expand(batch_shape + (-1,))
        event_shape = self.loc.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)
        if scale_tril is not None:
            self._unbroadcasted_scale_tril = scale_tril
        elif covariance_matrix is not None:
            self._unbroadcasted_scale_tril = torch.linalg.cholesky(covariance_matrix)
        else:
            self._unbroadcasted_scale_tril = _precision_to_scale_tril(precision_matrix)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MultivariateNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self.event_shape
        cov_shape = batch_shape + self.event_shape + self.event_shape
        new.loc = self.loc.expand(loc_shape)
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril
        if 'covariance_matrix' in self.__dict__:
            new.covariance_matrix = self.covariance_matrix.expand(cov_shape)
        if 'scale_tril' in self.__dict__:
            new.scale_tril = self.scale_tril.expand(cov_shape)
        if 'precision_matrix' in self.__dict__:
            new.precision_matrix = self.precision_matrix.expand(cov_shape)
        super(MultivariateNormal, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def scale_tril(self):
        return self._unbroadcasted_scale_tril.expand(self._batch_shape + self._event_shape + self._event_shape)

    @lazy_property
    def covariance_matrix(self):
        return torch.matmul(self._unbroadcasted_scale_tril, self._unbroadcasted_scale_tril.mT).expand(self._batch_shape + self._event_shape + self._event_shape)

    @lazy_property
    def precision_matrix(self):
        return torch.cholesky_inverse(self._unbroadcasted_scale_tril).expand(self._batch_shape + self._event_shape + self._event_shape)

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        return self._unbroadcasted_scale_tril.pow(2).sum(-1).expand(self._batch_shape + self._event_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + _batch_mv(self._unbroadcasted_scale_tril, eps)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M = _batch_mahalanobis(self._unbroadcasted_scale_tril, diff)
        half_log_det = self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M) - half_log_det

    def entropy(self):
        half_log_det = self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        H = 0.5 * self._event_shape[0] * (1.0 + math.log(2 * math.pi)) + half_log_det
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)