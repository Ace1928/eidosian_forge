from typing import Any, Callable, List, Optional, Tuple, Type, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.functional.classification.stat_scores import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import ClassificationTask
class MulticlassStatScores(_AbstractStatScores):
    """Computes true positives, false positives, true negatives, false negatives and the support for multiclass tasks.

    Related to `Type I and Type II errors`_.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)`` or float tensor of shape ``(N, C, ..)``.
      If preds is a floating point we apply ``torch.argmax`` along the ``C`` dimension to automatically convert
      probabilities/logits into an int tensor.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``


    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mcss`` (:class:`~torch.Tensor`): A tensor of shape ``(..., 5)``, where the last dimension corresponds
      to ``[tp, fp, tn, fn, sup]`` (``sup`` stands for support and equals ``tp + fn``). The shape
      depends on ``average`` and ``multidim_average`` parameters:

      - If ``multidim_average`` is set to ``global``:

        - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(5,)``
        - If ``average=None/'none'``, the shape will be ``(C, 5)``

      - If ``multidim_average`` is set to ``samplewise``:

        - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N, 5)``
        - If ``average=None/'none'``, the shape will be ``(N, C, 5)``

    If ``multidim_average`` is set to ``samplewise`` we expect at least one additional dimension ``...`` to be present,
    which the reduction will then be applied over instead of the sample dimension ``N``.

    Args:
        num_classes: Integer specifying the number of classes
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction
        top_k:
            Number of highest probability or logit score predictions considered to find the correct label.
            Only works when ``preds`` contain probabilities/logits.
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
        >>> from torchmetrics.classification import MulticlassStatScores
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> metric = MulticlassStatScores(num_classes=3, average='micro')
        >>> metric(preds, target)
        tensor([3, 1, 7, 1, 4])
        >>> mcss = MulticlassStatScores(num_classes=3, average=None)
        >>> mcss(preds, target)
        tensor([[1, 0, 2, 1, 2],
                [1, 1, 2, 0, 1],
                [1, 0, 3, 0, 1]])

    Example (preds is float tensor):
        >>> from torchmetrics.classification import MulticlassStatScores
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58],
        ...                 [0.22, 0.61, 0.17],
        ...                 [0.71, 0.09, 0.20],
        ...                 [0.05, 0.82, 0.13]])
        >>> metric = MulticlassStatScores(num_classes=3, average='micro')
        >>> metric(preds, target)
        tensor([3, 1, 7, 1, 4])
        >>> mcss = MulticlassStatScores(num_classes=3, average=None)
        >>> mcss(preds, target)
        tensor([[1, 0, 2, 1, 2],
                [1, 1, 2, 0, 1],
                [1, 0, 3, 0, 1]])

    Example (multidim tensors):
        >>> from torchmetrics.classification import MulticlassStatScores
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]])
        >>> metric = MulticlassStatScores(num_classes=3, multidim_average="samplewise", average='micro')
        >>> metric(preds, target)
        tensor([[3, 3, 9, 3, 6],
                [2, 4, 8, 4, 6]])
        >>> mcss = MulticlassStatScores(num_classes=3, multidim_average="samplewise", average=None)
        >>> mcss(preds, target)
        tensor([[[2, 1, 3, 0, 2],
                 [0, 1, 3, 2, 2],
                 [1, 1, 3, 1, 2]],
                [[0, 1, 4, 1, 1],
                 [1, 1, 2, 2, 3],
                 [1, 2, 2, 1, 2]]])

    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(self, num_classes: int, top_k: int=1, average: Optional[Literal['micro', 'macro', 'weighted', 'none']]='macro', multidim_average: Literal['global', 'samplewise']='global', ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> None:
        super(_AbstractStatScores, self).__init__(**kwargs)
        if validate_args:
            _multiclass_stat_scores_arg_validation(num_classes, top_k, average, multidim_average, ignore_index)
        self.num_classes = num_classes
        self.top_k = top_k
        self.average = average
        self.multidim_average = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self._create_state(size=1 if average == 'micro' and top_k == 1 else num_classes, multidim_average=multidim_average)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        if self.validate_args:
            _multiclass_stat_scores_tensor_validation(preds, target, self.num_classes, self.multidim_average, self.ignore_index)
        preds, target = _multiclass_stat_scores_format(preds, target, self.top_k)
        tp, fp, tn, fn = _multiclass_stat_scores_update(preds, target, self.num_classes, self.top_k, self.average, self.multidim_average, self.ignore_index)
        self._update_state(tp, fp, tn, fn)

    def compute(self) -> Tensor:
        """Compute the final statistics."""
        tp, fp, tn, fn = self._final_state()
        return _multiclass_stat_scores_compute(tp, fp, tn, fn, self.average, self.multidim_average)