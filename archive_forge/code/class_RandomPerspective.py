import math
import numbers
import warnings
from typing import Any, Callable, cast, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union
import PIL.Image
import torch
from torchvision import transforms as _transforms, tv_tensors
from torchvision.ops.boxes import box_iou
from torchvision.transforms.functional import _get_perspective_coeffs
from torchvision.transforms.v2 import functional as F, InterpolationMode, Transform
from torchvision.transforms.v2.functional._geometry import _check_interpolation
from torchvision.transforms.v2.functional._utils import _FillType
from ._transform import _RandomApplyTransform
from ._utils import (
class RandomPerspective(_RandomApplyTransform):
    """[BETA] Perform a random perspective transformation of the input with a given probability.

    .. v2betastatus:: RandomPerspective transform

    If the input is a :class:`torch.Tensor` or a ``TVTensor`` (e.g. :class:`~torchvision.tv_tensors.Image`,
    :class:`~torchvision.tv_tensors.Video`, :class:`~torchvision.tv_tensors.BoundingBoxes` etc.)
    it can have arbitrary number of leading batch dimensions. For example,
    the image can have ``[..., C, H, W]`` shape. A bounding box can have ``[..., 4]`` shape.

    Args:
        distortion_scale (float, optional): argument to control the degree of distortion and ranges from 0 to 1.
            Default is 0.5.
        p (float, optional): probability of the input being transformed. Default is 0.5.
        interpolation (InterpolationMode, optional): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        fill (number or tuple or dict, optional): Pixel fill value used when the  ``padding_mode`` is constant.
            Default is 0. If a tuple of length 3, it is used to fill R, G, B channels respectively.
            Fill value can be also a dictionary mapping data type to the fill value, e.g.
            ``fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}`` where ``Image`` will be filled with 127 and
            ``Mask`` will be filled with 0.
    """
    _v1_transform_cls = _transforms.RandomPerspective

    def __init__(self, distortion_scale: float=0.5, p: float=0.5, interpolation: Union[InterpolationMode, int]=InterpolationMode.BILINEAR, fill: Union[_FillType, Dict[Union[Type, str], _FillType]]=0) -> None:
        super().__init__(p=p)
        if not 0 <= distortion_scale <= 1:
            raise ValueError('Argument distortion_scale value should be between 0 and 1')
        self.distortion_scale = distortion_scale
        self.interpolation = _check_interpolation(interpolation)
        self.fill = fill
        self._fill = _setup_fill_arg(fill)

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        height, width = query_size(flat_inputs)
        distortion_scale = self.distortion_scale
        half_height = height // 2
        half_width = width // 2
        bound_height = int(distortion_scale * half_height) + 1
        bound_width = int(distortion_scale * half_width) + 1
        topleft = [int(torch.randint(0, bound_width, size=(1,))), int(torch.randint(0, bound_height, size=(1,)))]
        topright = [int(torch.randint(width - bound_width, width, size=(1,))), int(torch.randint(0, bound_height, size=(1,)))]
        botright = [int(torch.randint(width - bound_width, width, size=(1,))), int(torch.randint(height - bound_height, height, size=(1,)))]
        botleft = [int(torch.randint(0, bound_width, size=(1,))), int(torch.randint(height - bound_height, height, size=(1,)))]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        perspective_coeffs = _get_perspective_coeffs(startpoints, endpoints)
        return dict(coefficients=perspective_coeffs)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = _get_fill(self._fill, type(inpt))
        return self._call_kernel(F.perspective, inpt, None, None, fill=fill, interpolation=self.interpolation, **params)