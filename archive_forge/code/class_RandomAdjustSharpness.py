import collections.abc
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torchvision import transforms as _transforms
from torchvision.transforms.v2 import functional as F, Transform
from ._transform import _RandomApplyTransform
from ._utils import query_chw
class RandomAdjustSharpness(_RandomApplyTransform):
    """[BETA] Adjust the sharpness of the image or video with a given probability.

    .. v2betastatus:: RandomAdjustSharpness transform

    If the input is a :class:`torch.Tensor`,
    it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        sharpness_factor (float):  How much to adjust the sharpness. Can be
            any non-negative number. 0 gives a blurred image, 1 gives the
            original image while 2 increases the sharpness by a factor of 2.
        p (float): probability of the image being sharpened. Default value is 0.5
    """
    _v1_transform_cls = _transforms.RandomAdjustSharpness

    def __init__(self, sharpness_factor: float, p: float=0.5) -> None:
        super().__init__(p=p)
        self.sharpness_factor = sharpness_factor

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.adjust_sharpness, inpt, sharpness_factor=self.sharpness_factor)