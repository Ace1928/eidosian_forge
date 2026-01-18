import warnings
from .distance import PairwiseDistance
from .module import Module
from .. import functional as F
from .. import _reduction as _Reduction
from torch import Tensor
from typing import Callable, Optional
class NLLLoss2d(NLLLoss):

    def __init__(self, weight: Optional[Tensor]=None, size_average=None, ignore_index: int=-100, reduce=None, reduction: str='mean') -> None:
        warnings.warn('NLLLoss2d has been deprecated. Please use NLLLoss instead as a drop-in replacement and see https://pytorch.org/docs/master/nn.html#torch.nn.NLLLoss for more details.')
        super().__init__(weight, size_average, ignore_index, reduce, reduction)