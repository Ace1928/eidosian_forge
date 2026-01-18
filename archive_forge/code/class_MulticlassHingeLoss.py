from typing import Any, Optional, Sequence, Type, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.functional.classification.hinge import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
class MulticlassHingeLoss(Metric):
    """Compute the mean `Hinge loss`_ typically used for Support Vector Machines (SVMs) for multiclass tasks.

    The metric can be computed in two ways. Either, the definition by Crammer and Singer is used:

    .. math::
        \\text{Hinge loss} = \\max\\left(0, 1 - \\hat{y}_y + \\max_{i \\ne y} (\\hat{y}_i)\\right)

    Where :math:`y \\in {0, ..., \\mathrm{C}}` is the target class (where :math:`\\mathrm{C}` is the number of classes),
    and :math:`\\hat{y} \\in \\mathbb{R}^\\mathrm{C}` is the predicted output per class. Alternatively, the metric can
    also be computed in one-vs-all approach, where each class is valued against all other classes in a binary fashion.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A float tensor of shape ``(N, C, ...)``. Preds should be a tensor
      containing probabilities or logits for each observation. If preds has values outside [0,1] range we consider
      the input to be logits and will auto apply softmax per sample.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``. Target should be a tensor containing
      ground truth labels, and therefore only contain values in the [0, n_classes-1] range (except if `ignore_index`
      is specified).

    .. note::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``mchl`` (:class:`~torch.Tensor`): A tensor containing the multi-class hinge loss.

    Args:
        num_classes: Integer specifying the number of classes
        squared:
            If True, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss.
        multiclass_mode:
            Determines how to compute the metric
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics.classification import MulticlassHingeLoss
        >>> preds = torch.tensor([[0.25, 0.20, 0.55],
        ...                       [0.55, 0.05, 0.40],
        ...                       [0.10, 0.30, 0.60],
        ...                       [0.90, 0.05, 0.05]])
        >>> target = torch.tensor([0, 1, 2, 0])
        >>> mchl = MulticlassHingeLoss(num_classes=3)
        >>> mchl(preds, target)
        tensor(0.9125)
        >>> mchl = MulticlassHingeLoss(num_classes=3, squared=True)
        >>> mchl(preds, target)
        tensor(1.1131)
        >>> mchl = MulticlassHingeLoss(num_classes=3, multiclass_mode='one-vs-all')
        >>> mchl(preds, target)
        tensor([0.8750, 1.1250, 1.1000])

    """
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = 'Class'
    measures: Tensor
    total: Tensor

    def __init__(self, num_classes: int, squared: bool=False, multiclass_mode: Literal['crammer-singer', 'one-vs-all']='crammer-singer', ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multiclass_hinge_loss_arg_validation(num_classes, squared, multiclass_mode, ignore_index)
        self.validate_args = validate_args
        self.num_classes = num_classes
        self.squared = squared
        self.multiclass_mode = multiclass_mode
        self.ignore_index = ignore_index
        self.add_state('measures', default=torch.tensor(0.0) if self.multiclass_mode == 'crammer-singer' else torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric state."""
        if self.validate_args:
            _multiclass_hinge_loss_tensor_validation(preds, target, self.num_classes, self.ignore_index)
        preds, target = _multiclass_confusion_matrix_format(preds, target, self.ignore_index, convert_to_labels=False)
        measures, total = _multiclass_hinge_loss_update(preds, target, self.squared, self.multiclass_mode)
        self.measures += measures
        self.total += total

    def compute(self) -> Tensor:
        """Compute metric."""
        return _hinge_loss_compute(self.measures, self.total)

    def plot(self, val: Optional[Union[Tensor, Sequence[Tensor]]]=None, ax: Optional[_AX_TYPE]=None) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure object and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value per class
            >>> from torch import randint, randn
            >>> from torchmetrics.classification import MulticlassHingeLoss
            >>> metric = MulticlassHingeLoss(num_classes=3)
            >>> metric.update(randn(20, 3), randint(3, (20,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting a multiple values per class
            >>> from torch import randint, randn
            >>> from torchmetrics.classification import MulticlassHingeLoss
            >>> metric = MulticlassHingeLoss(num_classes=3)
            >>> values = []
            >>> for _ in range(20):
            ...     values.append(metric(randn(20, 3), randint(3, (20,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)