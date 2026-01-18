from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal
from torchmetrics.functional.image.utils import _gaussian_kernel_2d, _gaussian_kernel_3d, _reflection_pad_3d
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce
def _multiscale_ssim_update(preds: Tensor, target: Tensor, gaussian_kernel: bool=True, sigma: Union[float, Sequence[float]]=1.5, kernel_size: Union[int, Sequence[int]]=11, data_range: Optional[Union[float, Tuple[float, float]]]=None, k1: float=0.01, k2: float=0.03, betas: Union[Tuple[float, float, float, float, float], Tuple[float, ...]]=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333), normalize: Optional[Literal['relu', 'simple']]=None) -> Tensor:
    """Compute Multi-Scale Structural Similarity Index Measure.

    Adapted from: https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py.

    Args:
        preds: estimated image
        target: ground truth image
        gaussian_kernel: If true, a gaussian kernel is used, if false a uniform kernel is used
        sigma: Standard deviation of the gaussian kernel
        kernel_size: size of the gaussian kernel
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        data_range: Range of the image. If ``None``, it is determined from the image (max - min)
        k1: Parameter of structural similarity index measure.
        k2: Parameter of structural similarity index measure.
        betas: Exponent parameters for individual similarities and contrastive sensitives returned by different image
            resolutions.
        normalize: When MultiScaleSSIM loss is used for training, it is desirable to use normalizes to improve the
            training stability. This `normalize` argument is out of scope of the original implementation [1], and it is
            adapted from https://github.com/jorge-pessoa/pytorch-msssim instead.

    Raises:
        ValueError:
            If the image height or width is smaller then ``2 ** len(betas)``.
        ValueError:
            If the image height is smaller than ``(kernel_size[0] - 1) * max(1, (len(betas) - 1)) ** 2``.
        ValueError:
            If the image width is smaller than ``(kernel_size[0] - 1) * max(1, (len(betas) - 1)) ** 2``.

    """
    mcs_list: List[Tensor] = []
    is_3d = preds.ndim == 5
    if not isinstance(kernel_size, Sequence):
        kernel_size = 3 * [kernel_size] if is_3d else 2 * [kernel_size]
    if not isinstance(sigma, Sequence):
        sigma = 3 * [sigma] if is_3d else 2 * [sigma]
    if preds.size()[-1] < 2 ** len(betas) or preds.size()[-2] < 2 ** len(betas):
        raise ValueError(f'For a given number of `betas` parameters {len(betas)}, the image height and width dimensions must be larger than or equal to {2 ** len(betas)}.')
    _betas_div = max(1, len(betas) - 1) ** 2
    if preds.size()[-2] // _betas_div <= kernel_size[0] - 1:
        raise ValueError(f'For a given number of `betas` parameters {len(betas)} and kernel size {kernel_size[0]}, the image height must be larger than {(kernel_size[0] - 1) * _betas_div}.')
    if preds.size()[-1] // _betas_div <= kernel_size[1] - 1:
        raise ValueError(f'For a given number of `betas` parameters {len(betas)} and kernel size {kernel_size[1]}, the image width must be larger than {(kernel_size[1] - 1) * _betas_div}.')
    for _ in range(len(betas)):
        sim, contrast_sensitivity = _get_normalized_sim_and_cs(preds, target, gaussian_kernel, sigma, kernel_size, data_range, k1, k2, normalize=normalize)
        mcs_list.append(contrast_sensitivity)
        if len(kernel_size) == 2:
            preds = F.avg_pool2d(preds, (2, 2))
            target = F.avg_pool2d(target, (2, 2))
        elif len(kernel_size) == 3:
            preds = F.avg_pool3d(preds, (2, 2, 2))
            target = F.avg_pool3d(target, (2, 2, 2))
        else:
            raise ValueError('length of kernel_size is neither 2 nor 3')
    mcs_list[-1] = sim
    mcs_stack = torch.stack(mcs_list)
    if normalize == 'simple':
        mcs_stack = (mcs_stack + 1) / 2
    betas = torch.tensor(betas, device=mcs_stack.device).view(-1, 1)
    mcs_weighted = mcs_stack ** betas
    return torch.prod(mcs_weighted, axis=0)