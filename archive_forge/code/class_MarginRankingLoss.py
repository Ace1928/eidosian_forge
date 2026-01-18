import warnings
from .distance import PairwiseDistance
from .module import Module
from .. import functional as F
from .. import _reduction as _Reduction
from torch import Tensor
from typing import Callable, Optional
class MarginRankingLoss(_Loss):
    """Creates a criterion that measures the loss given
    inputs :math:`x1`, :math:`x2`, two 1D mini-batch or 0D `Tensors`,
    and a label 1D mini-batch or 0D `Tensor` :math:`y` (containing 1 or -1).

    If :math:`y = 1` then it assumed the first input should be ranked higher
    (have a larger value) than the second input, and vice-versa for :math:`y = -1`.

    The loss function for each pair of samples in the mini-batch is:

    .. math::
        \\text{loss}(x1, x2, y) = \\max(0, -y * (x1 - x2) + \\text{margin})

    Args:
        margin (float, optional): Has a default value of :math:`0`.
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
        - Input1: :math:`(N)` or :math:`()` where `N` is the batch size.
        - Input2: :math:`(N)` or :math:`()`, same shape as the Input1.
        - Target: :math:`(N)` or :math:`()`, same shape as the inputs.
        - Output: scalar. If :attr:`reduction` is ``'none'`` and Input size is not :math:`()`, then :math:`(N)`.

    Examples::

        >>> loss = nn.MarginRankingLoss()
        >>> input1 = torch.randn(3, requires_grad=True)
        >>> input2 = torch.randn(3, requires_grad=True)
        >>> target = torch.randn(3).sign()
        >>> output = loss(input1, input2, target)
        >>> output.backward()
    """
    __constants__ = ['margin', 'reduction']
    margin: float

    def __init__(self, margin: float=0.0, size_average=None, reduce=None, reduction: str='mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        return F.margin_ranking_loss(input1, input2, target, margin=self.margin, reduction=self.reduction)