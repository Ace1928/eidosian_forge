import math
import torch
import torch.jit
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, lazy_property
class VonMises(Distribution):
    """
    A circular von Mises distribution.

    This implementation uses polar coordinates. The ``loc`` and ``value`` args
    can be any real number (to facilitate unconstrained optimization), but are
    interpreted as angles modulo 2 pi.

    Example::
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = VonMises(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # von Mises distributed with loc=1 and concentration=1
        tensor([1.9777])

    :param torch.Tensor loc: an angle in radians.
    :param torch.Tensor concentration: concentration parameter
    """
    arg_constraints = {'loc': constraints.real, 'concentration': constraints.positive}
    support = constraints.real
    has_rsample = False

    def __init__(self, loc, concentration, validate_args=None):
        self.loc, self.concentration = broadcast_all(loc, concentration)
        batch_shape = self.loc.shape
        event_shape = torch.Size()
        super().__init__(batch_shape, event_shape, validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_prob = self.concentration * torch.cos(value - self.loc)
        log_prob = log_prob - math.log(2 * math.pi) - _log_modified_bessel_fn(self.concentration, order=0)
        return log_prob

    @lazy_property
    def _loc(self):
        return self.loc.to(torch.double)

    @lazy_property
    def _concentration(self):
        return self.concentration.to(torch.double)

    @lazy_property
    def _proposal_r(self):
        kappa = self._concentration
        tau = 1 + (1 + 4 * kappa ** 2).sqrt()
        rho = (tau - (2 * tau).sqrt()) / (2 * kappa)
        _proposal_r = (1 + rho ** 2) / (2 * rho)
        _proposal_r_taylor = 1 / kappa + kappa
        return torch.where(kappa < 1e-05, _proposal_r_taylor, _proposal_r)

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        """
        The sampling algorithm for the von Mises distribution is based on the
        following paper: D.J. Best and N.I. Fisher, "Efficient simulation of the
        von Mises distribution." Applied Statistics (1979): 152-157.

        Sampling is always done in double precision internally to avoid a hang
        in _rejection_sample() for small values of the concentration, which
        starts to happen for single precision around 1e-4 (see issue #88443).
        """
        shape = self._extended_shape(sample_shape)
        x = torch.empty(shape, dtype=self._loc.dtype, device=self.loc.device)
        return _rejection_sample(self._loc, self._concentration, self._proposal_r, x).to(self.loc.dtype)

    def expand(self, batch_shape):
        try:
            return super().expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            loc = self.loc.expand(batch_shape)
            concentration = self.concentration.expand(batch_shape)
            return type(self)(loc, concentration, validate_args=validate_args)

    @property
    def mean(self):
        """
        The provided mean is the circular one.
        """
        return self.loc

    @property
    def mode(self):
        return self.loc

    @lazy_property
    def variance(self):
        """
        The provided variance is the circular one.
        """
        return 1 - (_log_modified_bessel_fn(self.concentration, order=1) - _log_modified_bessel_fn(self.concentration, order=0)).exp()