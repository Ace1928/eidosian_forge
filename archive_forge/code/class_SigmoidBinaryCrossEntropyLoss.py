import numpy as np
from .. import ndarray
from ..base import numeric_types
from .block import HybridBlock
from ..util import is_np_array
class SigmoidBinaryCrossEntropyLoss(Loss):
    """The cross-entropy loss for binary classification. (alias: SigmoidBCELoss)

    BCE loss is useful when training logistic regression. If `from_sigmoid`
    is False (default), this loss computes:

    .. math::

        prob = \\frac{1}{1 + \\exp(-{pred})}

        L = - \\sum_i {label}_i * \\log({prob}_i) * pos\\_weight +
            (1 - {label}_i) * \\log(1 - {prob}_i)

    If `from_sigmoid` is True, this loss computes:

    .. math::

        L = - \\sum_i {label}_i * \\log({pred}_i) * pos\\_weight +
            (1 - {label}_i) * \\log(1 - {pred}_i)

    A tensor `pos_weight > 1` decreases the false negative count, hence increasing
    the recall.
    Conversely setting `pos_weight < 1` decreases the false positive count and
    increases the precision.

    `pred` and `label` can have arbitrary shape as long as they have the same
    number of elements.

    Parameters
    ----------
    from_sigmoid : bool, default is `False`
        Whether the input is from the output of sigmoid. Set this to false will make
        the loss calculate sigmoid and BCE together, which is more numerically
        stable through log-sum-exp trick.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: target tensor with values in range `[0, 1]`. Must have the
          same size as `pred`.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).
        - **pos_weight**: a weighting tensor of positive examples. Must be a vector with length
          equal to the number of classes.For example, if pred has shape (64, 10),
          pos_weight should have shape (1, 10).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, **kwargs):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__(weight, batch_axis, **kwargs)
        self._from_sigmoid = from_sigmoid

    def hybrid_forward(self, F, pred, label, sample_weight=None, pos_weight=None):
        label = _reshape_like(F, label, pred)
        if is_np_array():
            relu_fn = F.npx.relu
            act_fn = F.npx.activation
            abs_fn = F.np.abs
            mul_fn = F.np.multiply
            log_fn = F.np.log
        else:
            relu_fn = F.relu
            act_fn = F.Activation
            abs_fn = F.abs
            mul_fn = F.broadcast_mul
            log_fn = F.log
        if not self._from_sigmoid:
            if pos_weight is None:
                loss = relu_fn(pred) - pred * label + act_fn(-abs_fn(pred), act_type='softrelu')
            else:
                log_weight = 1 + mul_fn(pos_weight - 1, label)
                loss = pred - pred * label + log_weight * (act_fn(-abs_fn(pred), act_type='softrelu') + relu_fn(-pred))
        else:
            eps = 1e-12
            if pos_weight is None:
                loss = -(log_fn(pred + eps) * label + log_fn(1.0 - pred + eps) * (1.0 - label))
            else:
                loss = -(mul_fn(log_fn(pred + eps) * label, pos_weight) + log_fn(1.0 - pred + eps) * (1.0 - label))
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        if is_np_array():
            if F is ndarray:
                return F.np.mean(loss, axis=tuple(range(1, loss.ndim)))
            else:
                return F.npx.batch_flatten(loss).mean(axis=1)
        else:
            return F.mean(loss, axis=self._batch_axis, exclude=True)