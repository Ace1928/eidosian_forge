import numpy as np
from .. import ndarray
from ..base import numeric_types
from .block import HybridBlock
from ..util import is_np_array
def _apply_weighting(F, loss, weight=None, sample_weight=None):
    """Apply weighting to loss.

    Parameters
    ----------
    loss : Symbol
        The loss to be weighted.
    weight : float or None
        Global scalar weight for loss.
    sample_weight : Symbol or None
        Per sample weighting. Must be broadcastable to
        the same shape as loss. For example, if loss has
        shape (64, 10) and you want to weight each sample
        in the batch separately, `sample_weight` should have
        shape (64, 1).

    Returns
    -------
    loss : Symbol
        Weighted loss
    """
    if sample_weight is not None:
        if is_np_array():
            loss = loss * sample_weight
        else:
            loss = F.broadcast_mul(loss, sample_weight)
    if weight is not None:
        assert isinstance(weight, numeric_types), 'weight must be a number'
        loss = loss * weight
    return loss