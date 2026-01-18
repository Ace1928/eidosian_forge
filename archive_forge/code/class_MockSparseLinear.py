from torch.ao.pruning import BaseSparsifier
import torch
import torch.nn.functional as F
from torch import nn
class MockSparseLinear(nn.Linear):
    """
    This class is a MockSparseLinear class to check convert functionality.
    It is the same as a normal Linear layer, except with a different type, as
    well as an additional from_dense method.
    """

    @classmethod
    def from_dense(cls, mod):
        """
        """
        linear = cls(mod.in_features, mod.out_features)
        return linear