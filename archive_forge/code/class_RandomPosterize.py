import collections.abc
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torchvision import transforms as _transforms
from torchvision.transforms.v2 import functional as F, Transform
from ._transform import _RandomApplyTransform
from ._utils import query_chw
class RandomPosterize(_RandomApplyTransform):
    """[BETA] Posterize the image or video with a given probability by reducing the
    number of bits for each color channel.

    .. v2betastatus:: RandomPosterize transform

    If the input is a :class:`torch.Tensor`, it should be of type torch.uint8,
    and it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        bits (int): number of bits to keep for each channel (0-8)
        p (float): probability of the image being posterized. Default value is 0.5
    """
    _v1_transform_cls = _transforms.RandomPosterize

    def __init__(self, bits: int, p: float=0.5) -> None:
        super().__init__(p=p)
        self.bits = bits

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.posterize, inpt, bits=self.bits)