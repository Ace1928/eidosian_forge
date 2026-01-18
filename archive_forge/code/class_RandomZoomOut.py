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
class RandomZoomOut(_RandomApplyTransform):
    """[BETA] "Zoom out" transformation from
    `"SSD: Single Shot MultiBox Detector" <https://arxiv.org/abs/1512.02325>`_.

    .. v2betastatus:: RandomZoomOut transform

    This transformation randomly pads images, videos, bounding boxes and masks creating a zoom out effect.
    Output spatial size is randomly sampled from original size up to a maximum size configured
    with ``side_range`` parameter:

    .. code-block:: python

        r = uniform_sample(side_range[0], side_range[1])
        output_width = input_width * r
        output_height = input_height * r

    If the input is a :class:`torch.Tensor` or a ``TVTensor`` (e.g. :class:`~torchvision.tv_tensors.Image`,
    :class:`~torchvision.tv_tensors.Video`, :class:`~torchvision.tv_tensors.BoundingBoxes` etc.)
    it can have arbitrary number of leading batch dimensions. For example,
    the image can have ``[..., C, H, W]`` shape. A bounding box can have ``[..., 4]`` shape.

    Args:
        fill (number or tuple or dict, optional): Pixel fill value used when the  ``padding_mode`` is constant.
            Default is 0. If a tuple of length 3, it is used to fill R, G, B channels respectively.
            Fill value can be also a dictionary mapping data type to the fill value, e.g.
            ``fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}`` where ``Image`` will be filled with 127 and
            ``Mask`` will be filled with 0.
        side_range (sequence of floats, optional): tuple of two floats defines minimum and maximum factors to
            scale the input size.
        p (float, optional): probability that the zoom operation will be performed.
    """

    def __init__(self, fill: Union[_FillType, Dict[Union[Type, str], _FillType]]=0, side_range: Sequence[float]=(1.0, 4.0), p: float=0.5) -> None:
        super().__init__(p=p)
        self.fill = fill
        self._fill = _setup_fill_arg(fill)
        _check_sequence_input(side_range, 'side_range', req_sizes=(2,))
        self.side_range = side_range
        if side_range[0] < 1.0 or side_range[0] > side_range[1]:
            raise ValueError(f'Invalid canvas side range provided {side_range}.')

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        orig_h, orig_w = query_size(flat_inputs)
        r = self.side_range[0] + torch.rand(1) * (self.side_range[1] - self.side_range[0])
        canvas_width = int(orig_w * r)
        canvas_height = int(orig_h * r)
        r = torch.rand(2)
        left = int((canvas_width - orig_w) * r[0])
        top = int((canvas_height - orig_h) * r[1])
        right = canvas_width - (left + orig_w)
        bottom = canvas_height - (top + orig_h)
        padding = [left, top, right, bottom]
        return dict(padding=padding)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = _get_fill(self._fill, type(inpt))
        return self._call_kernel(F.pad, inpt, **params, fill=fill)