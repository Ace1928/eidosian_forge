import warnings
from torch import Tensor
from .batchnorm import _LazyNormBase, _NormBase
from .. import functional as F
def _apply_instance_norm(self, input):
    return F.instance_norm(input, self.running_mean, self.running_var, self.weight, self.bias, self.training or not self.track_running_stats, self.momentum, self.eps)