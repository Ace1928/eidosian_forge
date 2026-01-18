import torch
from torch.distributions import constraints
from torch.distributions.categorical import Categorical
from torch.distributions.distribution import Distribution
class OneHotCategoricalStraightThrough(OneHotCategorical):
    """
    Creates a reparameterizable :class:`OneHotCategorical` distribution based on the straight-
    through gradient estimator from [1].

    [1] Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation
    (Bengio et al, 2013)
    """
    has_rsample = True

    def rsample(self, sample_shape=torch.Size()):
        samples = self.sample(sample_shape)
        probs = self._categorical.probs
        return samples + (probs - probs.detach())