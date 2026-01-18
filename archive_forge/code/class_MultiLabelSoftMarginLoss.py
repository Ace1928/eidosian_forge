import warnings
from .distance import PairwiseDistance
from .module import Module
from .. import functional as F
from .. import _reduction as _Reduction
from torch import Tensor
from typing import Callable, Optional
class MultiLabelSoftMarginLoss(_WeightedLoss):
    """Creates a criterion that optimizes a multi-label one-versus-all
    loss based on max-entropy, between input :math:`x` and target :math:`y` of size
    :math:`(N, C)`.
    For each sample in the minibatch:

    .. math::
        loss(x, y) = - \\frac{1}{C} * \\sum_i y[i] * \\log((1 + \\exp(-x[i]))^{-1})
                         + (1-y[i]) * \\log\\left(\\frac{\\exp(-x[i])}{(1 + \\exp(-x[i]))}\\right)

    where :math:`i \\in \\left\\{0, \\; \\cdots , \\; \\text{x.nElement}() - 1\\right\\}`,
    :math:`y[i] \\in \\left\\{0, \\; 1\\right\\}`.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`. Otherwise, it is
            treated as if having all ones.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, C)` where `N` is the batch size and `C` is the number of classes.
        - Target: :math:`(N, C)`, label targets must have the same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(N)`.
    """
    __constants__ = ['reduction']

    def __init__(self, weight: Optional[Tensor]=None, size_average=None, reduce=None, reduction: str='mean') -> None:
        super().__init__(weight, size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.multilabel_soft_margin_loss(input, target, weight=self.weight, reduction=self.reduction)