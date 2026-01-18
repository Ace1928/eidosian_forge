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
class _TotalVariation(TotalVariation):
    """Wrapper for deprecated import.

    >>> import torch
    >>> _ = torch.manual_seed(42)
    >>> tv = _TotalVariation()
    >>> img = torch.rand(5, 3, 28, 28)
    >>> tv(img)
    tensor(7546.8018)

    """

    def __init__(self, reduction: Literal['mean', 'sum', 'none', None]='sum', **kwargs: Any) -> None:
        _deprecated_root_import_class('TotalVariation', 'image')
        super().__init__(reduction=reduction, **kwargs)