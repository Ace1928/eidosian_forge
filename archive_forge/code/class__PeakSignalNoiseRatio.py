from typing import Any, Dict, Optional, Sequence, Tuple, Union
from typing_extensions import Literal
from torchmetrics.image.d_lambda import SpectralDistortionIndex
from torchmetrics.image.ergas import ErrorRelativeGlobalDimensionlessSynthesis
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.rase import RelativeAverageSpectralError
from torchmetrics.image.rmse_sw import RootMeanSquaredErrorUsingSlidingWindow
from torchmetrics.image.sam import SpectralAngleMapper
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure, StructuralSimilarityIndexMeasure
from torchmetrics.image.tv import TotalVariation
from torchmetrics.image.uqi import UniversalImageQualityIndex
from torchmetrics.utilities.prints import _deprecated_root_import_class
class _PeakSignalNoiseRatio(PeakSignalNoiseRatio):
    """Wrapper for deprecated import.

    >>> from torch import tensor
    >>> psnr = _PeakSignalNoiseRatio()
    >>> preds = tensor([[0.0, 1.0], [2.0, 3.0]])
    >>> target = tensor([[3.0, 2.0], [1.0, 0.0]])
    >>> psnr(preds, target)
    tensor(2.5527)

    """

    def __init__(self, data_range: Optional[Union[float, Tuple[float, float]]]=None, base: float=10.0, reduction: Literal['elementwise_mean', 'sum', 'none', None]='elementwise_mean', dim: Optional[Union[int, Tuple[int, ...]]]=None, **kwargs: Any) -> None:
        _deprecated_root_import_class('PeakSignalNoiseRatio', 'image')
        super().__init__(data_range=data_range, base=base, reduction=reduction, dim=dim, **kwargs)