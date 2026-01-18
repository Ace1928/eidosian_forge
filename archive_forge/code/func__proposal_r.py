import math
import torch
import torch.jit
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all, lazy_property
@lazy_property
def _proposal_r(self):
    kappa = self._concentration
    tau = 1 + (1 + 4 * kappa ** 2).sqrt()
    rho = (tau - (2 * tau).sqrt()) / (2 * kappa)
    _proposal_r = (1 + rho ** 2) / (2 * rho)
    _proposal_r_taylor = 1 / kappa + kappa
    return torch.where(kappa < 1e-05, _proposal_r_taylor, _proposal_r)