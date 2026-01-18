import warnings
from torch import Tensor
from .batchnorm import _LazyNormBase, _NormBase
from .. import functional as F
def _handle_no_batch_input(self, input):
    return self._apply_instance_norm(input.unsqueeze(0)).squeeze(0)