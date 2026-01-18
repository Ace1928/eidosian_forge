import warnings
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Type, Union
import PIL.Image
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import transforms as _transforms, tv_tensors
from torchvision.transforms.v2 import functional as F, Transform
from ._utils import _parse_labels_getter, _setup_number_or_seq, _setup_size, get_bounding_boxes, has_any, is_pure_tensor
class GaussianBlur(Transform):
    """[BETA] Blurs image with randomly chosen Gaussian blur.

    .. v2betastatus:: GausssianBlur transform

    If the input is a Tensor, it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.
    """
    _v1_transform_cls = _transforms.GaussianBlur

    def __init__(self, kernel_size: Union[int, Sequence[int]], sigma: Union[int, float, Sequence[float]]=(0.1, 2.0)) -> None:
        super().__init__()
        self.kernel_size = _setup_size(kernel_size, 'Kernel size should be a tuple/list of two integers')
        for ks in self.kernel_size:
            if ks <= 0 or ks % 2 == 0:
                raise ValueError('Kernel size value should be an odd and positive number.')
        self.sigma = _setup_number_or_seq(sigma, 'sigma')
        if not 0.0 < self.sigma[0] <= self.sigma[1]:
            raise ValueError(f'sigma values should be positive and of the form (min, max). Got {self.sigma}')

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sigma = torch.empty(1).uniform_(self.sigma[0], self.sigma[1]).item()
        return dict(sigma=[sigma, sigma])

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.gaussian_blur, inpt, self.kernel_size, **params)