import torch
from torch.distributions.distribution import Distribution
@property
def _mean_carrier_measure(self):
    """
        Abstract method for expected carrier measure, which is required for computing
        entropy.
        """
    raise NotImplementedError