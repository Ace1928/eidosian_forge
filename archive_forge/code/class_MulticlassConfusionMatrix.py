from typing import Any, List, Optional, Type
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.functional.classification.confusion_matrix import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE, plot_confusion_matrix
class MulticlassConfusionMatrix(Metric):
    """Compute the `confusion matrix`_ for multiclass tasks.

    The confusion matrix :math:`C` is constructed such that :math:`C_{i, j}` is equal to the number of observations
    known to be in class :math:`i` but predicted to be in class :math:`j`. Thus row indices of the confusion matrix
    correspond to the true class labels and column indices correspond to the predicted class labels.

    For multiclass tasks, the confusion matrix is a NxN matrix, where:

    - :math:`C_{i, i}` represents the number of true positives for class :math:`i`
    - :math:`\\sum_{j=1, j\\neq i}^N C_{i, j}` represents the number of false negatives for class :math:`i`
    - :math:`\\sum_{i=1, i\\neq j}^N C_{i, j}` represents the number of false positives for class :math:`i`
    - the sum of the remaining cells in the matrix represents the number of true negatives for class :math:`i`

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): An int or float tensor of shape ``(N, ...)``. If preds is a floating point
      tensor with values outside [0,1] range we consider the input to be logits and will auto apply sigmoid per
      element. Additionally, we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``confusion_matrix``: [num_classes, num_classes] matrix

    Args:
        num_classes: Integer specifying the number of classes
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        normalize: Normalization mode for confusion matrix. Choose from:

            - ``None`` or ``'none'``: no normalization (default)
            - ``'true'``: normalization over the targets (most commonly used)
            - ``'pred'``: normalization over the predictions
            - ``'all'``: normalization over the whole matrix
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (pred is integer tensor):
        >>> from torch import tensor
        >>> from torchmetrics.classification import MulticlassConfusionMatrix
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> metric = MulticlassConfusionMatrix(num_classes=3)
        >>> metric(preds, target)
        tensor([[1, 1, 0],
                [0, 1, 0],
                [0, 0, 1]])

    Example (pred is float tensor):
        >>> from torchmetrics.classification import MulticlassConfusionMatrix
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58],
        ...                 [0.22, 0.61, 0.17],
        ...                 [0.71, 0.09, 0.20],
        ...                 [0.05, 0.82, 0.13]])
        >>> metric = MulticlassConfusionMatrix(num_classes=3)
        >>> metric(preds, target)
        tensor([[1, 1, 0],
                [0, 1, 0],
                [0, 0, 1]])

    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False
    confmat: Tensor

    def __init__(self, num_classes: int, ignore_index: Optional[int]=None, normalize: Optional[Literal['none', 'true', 'pred', 'all']]=None, validate_args: bool=True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multiclass_confusion_matrix_arg_validation(num_classes, ignore_index, normalize)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.normalize = normalize
        self.validate_args = validate_args
        self.add_state('confmat', torch.zeros(num_classes, num_classes, dtype=torch.long), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        if self.validate_args:
            _multiclass_confusion_matrix_tensor_validation(preds, target, self.num_classes, self.ignore_index)
        preds, target = _multiclass_confusion_matrix_format(preds, target, self.ignore_index)
        confmat = _multiclass_confusion_matrix_update(preds, target, self.num_classes)
        self.confmat += confmat

    def compute(self) -> Tensor:
        """Compute confusion matrix."""
        return _multiclass_confusion_matrix_compute(self.confmat, self.normalize)

    def plot(self, val: Optional[Tensor]=None, ax: Optional[_AX_TYPE]=None, add_text: bool=True, labels: Optional[List[str]]=None) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis
            add_text: if the value of each cell should be added to the plot
            labels: a list of strings, if provided will be added to the plot to indicate the different classes

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> from torch import randint
            >>> from torchmetrics.classification import MulticlassConfusionMatrix
            >>> metric = MulticlassConfusionMatrix(num_classes=5)
            >>> metric.update(randint(5, (20,)), randint(5, (20,)))
            >>> fig_, ax_ = metric.plot()

        """
        val = val if val is not None else self.compute()
        if not isinstance(val, Tensor):
            raise TypeError(f'Expected val to be a single tensor but got {val}')
        fig, ax = plot_confusion_matrix(val, ax=ax, add_text=add_text, labels=labels)
        return (fig, ax)