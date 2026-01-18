from typing import Dict
import torch
from torch.distributions import Categorical, constraints
from torch.distributions.distribution import Distribution
def _pad_mixture_dimensions(self, x):
    dist_batch_ndims = self.batch_shape.numel()
    cat_batch_ndims = self.mixture_distribution.batch_shape.numel()
    pad_ndims = 0 if cat_batch_ndims == 1 else dist_batch_ndims - cat_batch_ndims
    xs = x.shape
    x = x.reshape(xs[:-1] + torch.Size(pad_ndims * [1]) + xs[-1:] + torch.Size(self._event_ndims * [1]))
    return x