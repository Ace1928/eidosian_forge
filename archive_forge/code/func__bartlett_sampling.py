import math
import warnings
from numbers import Number
from typing import Optional, Union
import torch
from torch import nan
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.multivariate_normal import _precision_to_scale_tril
from torch.distributions.utils import lazy_property
def _bartlett_sampling(self, sample_shape=torch.Size()):
    p = self._event_shape[-1]
    noise = _clamp_above_eps(self._dist_chi2.rsample(sample_shape).sqrt()).diag_embed(dim1=-2, dim2=-1)
    i, j = torch.tril_indices(p, p, offset=-1)
    noise[..., i, j] = torch.randn(torch.Size(sample_shape) + self._batch_shape + (int(p * (p - 1) / 2),), dtype=noise.dtype, device=noise.device)
    chol = self._unbroadcasted_scale_tril @ noise
    return chol @ chol.transpose(-2, -1)