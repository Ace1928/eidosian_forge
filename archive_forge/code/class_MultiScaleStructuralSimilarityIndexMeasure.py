from typing import Any, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal
from torchmetrics.functional.image.ssim import _multiscale_ssim_update, _ssim_check_inputs, _ssim_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
class MultiScaleStructuralSimilarityIndexMeasure(Metric):
    """Compute `MultiScaleSSIM`_, Multi-scale Structural Similarity Index Measure.

    This metric is is a generalization of Structural Similarity Index Measure by incorporating image details at
    different resolution scores.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): Predictions from model
    - ``target`` (:class:`~torch.Tensor`): Ground truth values

    As output of `forward` and `compute` the metric returns the following output

    - ``msssim`` (:class:`~torch.Tensor`): if ``reduction!='none'`` returns float scalar tensor with average MSSSIM
      value over sample else returns tensor of shape ``(N,)`` with SSIM values per sample

    Args:
        gaussian_kernel: If ``True`` (default), a gaussian kernel is used, if false a uniform kernel is used
        kernel_size: size of the gaussian kernel
        sigma: Standard deviation of the gaussian kernel
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        data_range:
            the range of the data. If None, it is determined from the data (max - min). If a tuple is provided then
            the range is calculated as the difference and input is clamped between the values.
            The ``data_range`` must be given when ``dim`` is not None.
        k1: Parameter of structural similarity index measure.
        k2: Parameter of structural similarity index measure.
        betas: Exponent parameters for individual similarities and contrastive sensitivities returned by different image
            resolutions.
        normalize: When MultiScaleStructuralSimilarityIndexMeasure loss is used for training, it is desirable to use
            normalizes to improve the training stability. This `normalize` argument is out of scope of the original
            implementation [1], and it is adapted from https://github.com/jorge-pessoa/pytorch-msssim instead.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Return:
        Tensor with Multi-Scale SSIM score

    Raises:
        ValueError:
            If ``kernel_size`` is not an int or a Sequence of ints with size 2 or 3.
        ValueError:
            If ``betas`` is not a tuple of floats with length 2.
        ValueError:
            If ``normalize`` is neither `None`, `ReLU` nor `simple`.

    Example:
        >>> from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
        >>> import torch
        >>> gen = torch.manual_seed(42)
        >>> preds = torch.rand([3, 3, 256, 256], generator=torch.manual_seed(42))
        >>> target = preds * 0.75
        >>> ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
        >>> ms_ssim(preds, target)
        tensor(0.9627)

    """
    higher_is_better: bool = True
    is_differentiable: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(self, gaussian_kernel: bool=True, kernel_size: Union[int, Sequence[int]]=11, sigma: Union[float, Sequence[float]]=1.5, reduction: Literal['elementwise_mean', 'sum', 'none', None]='elementwise_mean', data_range: Optional[Union[float, Tuple[float, float]]]=None, k1: float=0.01, k2: float=0.03, betas: Tuple[float, ...]=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333), normalize: Literal['relu', 'simple', None]='relu', **kwargs: Any) -> None:
        super().__init__(**kwargs)
        valid_reduction = ('elementwise_mean', 'sum', 'none', None)
        if reduction not in valid_reduction:
            raise ValueError(f'Argument `reduction` must be one of {valid_reduction}, but got {reduction}')
        if reduction in ('elementwise_mean', 'sum'):
            self.add_state('similarity', default=torch.tensor(0.0), dist_reduce_fx='sum')
        else:
            self.add_state('similarity', default=[], dist_reduce_fx='cat')
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')
        if not isinstance(kernel_size, (Sequence, int)):
            raise ValueError(f'Argument `kernel_size` expected to be an sequence or an int, or a single int. Got {kernel_size}')
        if isinstance(kernel_size, Sequence) and (len(kernel_size) not in (2, 3) or not all((isinstance(ks, int) for ks in kernel_size))):
            raise ValueError(f'Argument `kernel_size` expected to be an sequence of size 2 or 3 where each element is an int, or a single int. Got {kernel_size}')
        self.gaussian_kernel = gaussian_kernel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        if not isinstance(betas, tuple):
            raise ValueError('Argument `betas` is expected to be of a type tuple.')
        if isinstance(betas, tuple) and (not all((isinstance(beta, float) for beta in betas))):
            raise ValueError('Argument `betas` is expected to be a tuple of floats.')
        self.betas = betas
        if normalize and normalize not in ('relu', 'simple'):
            raise ValueError("Argument `normalize` to be expected either `None` or one of 'relu' or 'simple'")
        self.normalize = normalize

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        preds, target = _ssim_check_inputs(preds, target)
        similarity = _multiscale_ssim_update(preds, target, self.gaussian_kernel, self.sigma, self.kernel_size, self.data_range, self.k1, self.k2, self.betas, self.normalize)
        if self.reduction in ('none', None):
            self.similarity.append(similarity)
        else:
            self.similarity += similarity.sum()
        self.total += preds.shape[0]

    def compute(self) -> Tensor:
        """Compute MS-SSIM over state."""
        if self.reduction in ('none', None):
            return dim_zero_cat(self.similarity)
        if self.reduction == 'sum':
            return self.similarity
        return self.similarity / self.total

    def plot(self, val: Optional[Union[Tensor, Sequence[Tensor]]]=None, ax: Optional[_AX_TYPE]=None) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
            >>> import torch
            >>> preds = torch.rand([3, 3, 256, 256], generator=torch.manual_seed(42))
            >>> target = preds * 0.75
            >>> metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
            >>> import torch
            >>> preds = torch.rand([3, 3, 256, 256], generator=torch.manual_seed(42))
            >>> target = preds * 0.75
            >>> metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(preds, target))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)