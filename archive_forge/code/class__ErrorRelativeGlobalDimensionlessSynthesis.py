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
class _ErrorRelativeGlobalDimensionlessSynthesis(ErrorRelativeGlobalDimensionlessSynthesis):
    """Wrapper for deprecated import.

    >>> import torch
    >>> preds = torch.rand([16, 1, 16, 16], generator=torch.manual_seed(42))
    >>> target = preds * 0.75
    >>> ergas = _ErrorRelativeGlobalDimensionlessSynthesis()
    >>> torch.round(ergas(preds, target))
    tensor(154.)

    """

    def __init__(self, ratio: float=4, reduction: Literal['elementwise_mean', 'sum', 'none', None]='elementwise_mean', **kwargs: Any) -> None:
        _deprecated_root_import_class('ErrorRelativeGlobalDimensionlessSynthesis', 'image')
        super().__init__(ratio=ratio, reduction=reduction, **kwargs)