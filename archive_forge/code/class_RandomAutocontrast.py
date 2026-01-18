import collections.abc
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torchvision import transforms as _transforms
from torchvision.transforms.v2 import functional as F, Transform
from ._transform import _RandomApplyTransform
from ._utils import query_chw
class RandomAutocontrast(_RandomApplyTransform):
    """[BETA] Autocontrast the pixels of the given image or video with a given probability.

    .. v2betastatus:: RandomAutocontrast transform

    If the input is a :class:`torch.Tensor`, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        p (float): probability of the image being autocontrasted. Default value is 0.5
    """
    _v1_transform_cls = _transforms.RandomAutocontrast

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.autocontrast, inpt)