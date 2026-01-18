import warnings
from torch import Tensor
from .batchnorm import _LazyNormBase, _NormBase
from .. import functional as F
def _get_no_batch_dim(self):
    return 4