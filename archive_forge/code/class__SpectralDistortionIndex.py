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
class _SpectralDistortionIndex(SpectralDistortionIndex):
    """Wrapper for deprecated import.

    >>> import torch
    >>> _ = torch.manual_seed(42)
    >>> preds = torch.rand([16, 3, 16, 16])
    >>> target = torch.rand([16, 3, 16, 16])
    >>> sdi = _SpectralDistortionIndex()
    >>> sdi(preds, target)
    tensor(0.0234)

    """

    def __init__(self, p: int=1, reduction: Literal['elementwise_mean', 'sum', 'none']='elementwise_mean', **kwargs: Any) -> None:
        _deprecated_root_import_class('SpectralDistortionIndex', 'image')
        super().__init__(p=p, reduction=reduction, **kwargs)