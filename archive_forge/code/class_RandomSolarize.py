import collections.abc
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torchvision import transforms as _transforms
from torchvision.transforms.v2 import functional as F, Transform
from ._transform import _RandomApplyTransform
from ._utils import query_chw
class RandomSolarize(_RandomApplyTransform):
    """[BETA] Solarize the image or video with a given probability by inverting all pixel
    values above a threshold.

    .. v2betastatus:: RandomSolarize transform

    If img is a Tensor, it is expected to be in [..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        threshold (float): all pixels equal or above this value are inverted.
        p (float): probability of the image being solarized. Default value is 0.5
    """
    _v1_transform_cls = _transforms.RandomSolarize

    def __init__(self, threshold: float, p: float=0.5) -> None:
        super().__init__(p=p)
        self.threshold = threshold

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.solarize, inpt, threshold=self.threshold)