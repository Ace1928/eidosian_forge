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
class BinaryHingeLoss(Metric):
    """Compute the mean `Hinge loss`_ typically used for Support Vector Machines (SVMs) for binary tasks.

    .. math::
        \\text{Hinge loss} = \\max(0, 1 - y \\times \\hat{y})

    Where :math:`y \\in {-1, 1}` is the target, and :math:`\\hat{y} \\in \\mathbb{R}` is the prediction.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A float tensor of shape ``(N, ...)``. Preds should be a tensor containing
      probabilities or logits for each observation. If preds has values outside [0,1] range we consider the input
      to be logits and will auto apply sigmoid per element.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``. Target should be a tensor containing
      ground truth labels, and therefore only contain {0,1} values (except if `ignore_index` is specified). The value
      1 always encodes the positive class.

    .. note::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``bhl`` (:class:`~torch.Tensor`): A tensor containing the hinge loss.

    Args:
        squared:
            If True, this will compute the squared hinge loss. Otherwise, computes the regular hinge loss.
        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics.classification import BinaryHingeLoss
        >>> preds = torch.tensor([0.25, 0.25, 0.55, 0.75, 0.75])
        >>> target = torch.tensor([0, 0, 1, 1, 1])
        >>> bhl = BinaryHingeLoss()
        >>> bhl(preds, target)
        tensor(0.6900)
        >>> bhl = BinaryHingeLoss(squared=True)
        >>> bhl(preds, target)
        tensor(0.6905)

    """
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    measures: Tensor
    total: Tensor

    def __init__(self, squared: bool=False, ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _binary_hinge_loss_arg_validation(squared, ignore_index)
        self.validate_args = validate_args
        self.squared = squared
        self.ignore_index = ignore_index
        self.add_state('measures', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric state."""
        if self.validate_args:
            _binary_hinge_loss_tensor_validation(preds, target, self.ignore_index)
        preds, target = _binary_confusion_matrix_format(preds, target, threshold=0.0, ignore_index=self.ignore_index, convert_to_labels=False)
        measures, total = _binary_hinge_loss_update(preds, target, self.squared)
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

            >>> # Example plotting a single value
            >>> from torch import rand, randint
            >>> from torchmetrics.classification import BinaryHingeLoss
            >>> metric = BinaryHingeLoss()
            >>> metric.update(rand(10), randint(2,(10,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from torch import rand, randint
            >>> from torchmetrics.classification import BinaryHingeLoss
            >>> metric = BinaryHingeLoss()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(rand(10), randint(2,(10,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)