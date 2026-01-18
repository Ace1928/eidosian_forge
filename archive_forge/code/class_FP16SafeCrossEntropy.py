import math
from itertools import chain
from typing import Optional
import parlai.utils.logging as logging
from parlai.utils.misc import error_once
class FP16SafeCrossEntropy(torch.nn.Module):
    """
    FP16-safe cross entropy loss.

    This avoids overflow in the softmax by doing the operation in FP32.
    """

    def __init__(self, weight: Optional[torch.Tensor]=None, ignore_index: int=-100, reduction: str='none'):
        super().__init__()
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, scores, targets):
        return F.nll_loss(F.log_softmax(scores, 1, dtype=torch.float32), targets, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)