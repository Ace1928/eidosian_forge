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
class _MultiScaleStructuralSimilarityIndexMeasure(MultiScaleStructuralSimilarityIndexMeasure):
    """Wrapper for deprecated import.

    >>> import torch
    >>> preds = torch.rand([3, 3, 256, 256], generator=torch.manual_seed(42))
    >>> target = preds * 0.75
    >>> ms_ssim = _MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
    >>> ms_ssim(preds, target)
    tensor(0.9627)

    """

    def __init__(self, gaussian_kernel: bool=True, kernel_size: Union[int, Sequence[int]]=11, sigma: Union[float, Sequence[float]]=1.5, reduction: Literal['elementwise_mean', 'sum', 'none', None]='elementwise_mean', data_range: Optional[Union[float, Tuple[float, float]]]=None, k1: float=0.01, k2: float=0.03, betas: Tuple[float, ...]=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333), normalize: Literal['relu', 'simple', None]='relu', **kwargs: Any) -> None:
        _deprecated_root_import_class('MultiScaleStructuralSimilarityIndexMeasure', 'image')
        super().__init__(gaussian_kernel=gaussian_kernel, kernel_size=kernel_size, sigma=sigma, reduction=reduction, data_range=data_range, k1=k1, k2=k2, betas=betas, normalize=normalize, **kwargs)