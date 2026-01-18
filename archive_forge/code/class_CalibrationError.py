from typing import Any, List, Optional, Sequence, Type, Union
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.functional.classification.calibration_error import (
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
class CalibrationError(_ClassificationTaskWrapper):
    """`Top-label Calibration Error`_.

    The expected calibration error can be used to quantify how well a given model is calibrated e.g. how well the
    predicted output probabilities of the model matches the actual probabilities of the ground truth distribution.
    Three different norms are implemented, each corresponding to variations on the calibration error metric.

    .. math::
        \\text{ECE} = \\sum_i^N b_i \\|(p_i - c_i)\\|, \\text{L1 norm (Expected Calibration Error)}

    .. math::
        \\text{MCE} =  \\max_{i} (p_i - c_i), \\text{Infinity norm (Maximum Calibration Error)}

    .. math::
        \\text{RMSCE} = \\sqrt{\\sum_i^N b_i(p_i - c_i)^2}, \\text{L2 norm (Root Mean Square Calibration Error)}

    Where :math:`p_i` is the top-1 prediction accuracy in bin :math:`i`, :math:`c_i` is the average confidence of
    predictions in bin :math:`i`, and :math:`b_i` is the fraction of data points in bin :math:`i`. Bins are constructed
    in an uniform way in the [0,1] range.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'`` or ``'multiclass'``. See the documentation of
    :class:`~torchmetrics.classification.BinaryCalibrationError` and
    :class:`~torchmetrics.classification.MulticlassCalibrationError` for the specific details of each argument influence
    and examples.

    """

    def __new__(cls: Type['CalibrationError'], task: Literal['binary', 'multiclass'], n_bins: int=15, norm: Literal['l1', 'l2', 'max']='l1', num_classes: Optional[int]=None, ignore_index: Optional[int]=None, validate_args: bool=True, **kwargs: Any) -> Metric:
        """Initialize task metric."""
        task = ClassificationTaskNoMultilabel.from_str(task)
        kwargs.update({'n_bins': n_bins, 'norm': norm, 'ignore_index': ignore_index, 'validate_args': validate_args})
        if task == ClassificationTaskNoMultilabel.BINARY:
            return BinaryCalibrationError(**kwargs)
        if task == ClassificationTaskNoMultilabel.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f'`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`')
            return MulticlassCalibrationError(num_classes, **kwargs)
        raise ValueError(f'Not handled value: {task}')