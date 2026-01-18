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
class _RootMeanSquaredErrorUsingSlidingWindow(RootMeanSquaredErrorUsingSlidingWindow):
    """Wrapper for deprecated import.

    >>> import torch
    >>> g = torch.manual_seed(22)
    >>> preds = torch.rand(4, 3, 16, 16)
    >>> target = torch.rand(4, 3, 16, 16)
    >>> rmse_sw = RootMeanSquaredErrorUsingSlidingWindow()
    >>> rmse_sw(preds, target)
    tensor(0.3999)

    """

    def __init__(self, window_size: int=8, **kwargs: Dict[str, Any]) -> None:
        _deprecated_root_import_class('RootMeanSquaredErrorUsingSlidingWindow', 'image')
        super().__init__(window_size=window_size, **kwargs)