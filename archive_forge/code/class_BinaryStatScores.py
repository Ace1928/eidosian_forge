from typing import Any, Callable, List, Optional, Tuple, Type, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import ClassificationTask
class BinaryStatScores(_AbstractStatScores):
    """Compute true positives, false positives, true negatives, false negatives and the support for binary tasks.

    Related to `Type I and Type II errors`_.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): An int or float tensor of shape ``(N, ...)``. If preds is a floating
      point tensor with values outside [0,1] range we consider the input to be logits and will auto apply sigmoid
      per element. Additionally, we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``


    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``bss`` (:class:`~torch.Tensor`): A tensor of shape ``(..., 5)``, where the last dimension corresponds
      to ``[tp, fp, tn, fn, sup]`` (``sup`` stands for support and equals ``tp + fn``). The shape
      depends on the ``multidim_average`` parameter:

      - If ``multidim_average`` is set to ``global``, the shape will be ``(5,)``
      - If ``multidim_average`` is set to ``samplewise``, the shape will be ``(N, 5)``

    If ``multidim_average`` is set to ``samplewise`` we expect at least one additional dimension ``...`` to be present,
    which the reduction will then be applied over instead of the sample dimension ``N``.

    Args:
        threshold: Threshold for transforming probability to binary {0,1} predictions
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.classification import BinaryStatScores
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0, 0, 1, 1, 0, 1])
        >>> metric = BinaryStatScores()
        >>> metric(preds, target)
        tensor([2, 1, 2, 1, 3])

    Example (preds is float tensor):
        >>> from torchmetrics.classification import BinaryStatScores
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
        >>> metric = BinaryStatScores()
        >>> metric(preds, target)
        tensor([2, 1, 2, 1, 3])

    Example (multidim tensors):
        >>> from torchmetrics.classification import BinaryStatScores
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
        ...                 [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
        >>> metric = BinaryStatScores(multidim_average='samplewise')
        >>> metric(preds, target)
        tensor([[2, 3, 0, 1, 3],
                [0, 2, 1, 3, 3]])

    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(self, threshold: float=0.5, multidim_average: Literal['global', 'samplewise']='global', ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> None:
        super(_AbstractStatScores, self).__init__(**kwargs)
        if validate_args:
            _binary_stat_scores_arg_validation(threshold, multidim_average, ignore_index)
        self.threshold = threshold
        self.multidim_average = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self._create_state(size=1, multidim_average=multidim_average)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        if self.validate_args:
            _binary_stat_scores_tensor_validation(preds, target, self.multidim_average, self.ignore_index)
        preds, target = _binary_stat_scores_format(preds, target, self.threshold, self.ignore_index)
        tp, fp, tn, fn = _binary_stat_scores_update(preds, target, self.multidim_average)
        self._update_state(tp, fp, tn, fn)

    def compute(self) -> Tensor:
        """Compute the final statistics."""
        tp, fp, tn, fn = self._final_state()
        return _binary_stat_scores_compute(tp, fp, tn, fn, self.multidim_average)