import numpy as np
from .. import ndarray
from ..base import numeric_types
from .block import HybridBlock
from ..util import is_np_array
class SquaredHingeLoss(Loss):
    """Calculates the soft-margin loss function used in SVMs:

    .. math::
        L = \\sum_i max(0, {margin} - {pred}_i \\cdot {label}_i)^2

    where `pred` is the classifier prediction and `label` is the target tensor
    containing values -1 or 1. `label` and `pred` can have arbitrary shape as
    long as they have the same number of elements.

    Parameters
    ----------
    margin : float
        The margin in hinge loss. Defaults to 1.0
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: truth tensor with values -1 or 1. Must have the same size
          as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, margin=1, weight=None, batch_axis=0, **kwargs):
        super(SquaredHingeLoss, self).__init__(weight, batch_axis, **kwargs)
        self._margin = margin

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        label = _reshape_like(F, label, pred)
        loss = F.square(F.relu(self._margin - pred * label))
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)