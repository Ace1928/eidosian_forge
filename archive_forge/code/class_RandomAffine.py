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
class RandomAffine(Transform):
    """[BETA] Random affine transformation the input keeping center invariant.

    .. v2betastatus:: RandomAffine transform

    If the input is a :class:`torch.Tensor` or a ``TVTensor`` (e.g. :class:`~torchvision.tv_tensors.Image`,
    :class:`~torchvision.tv_tensors.Video`, :class:`~torchvision.tv_tensors.BoundingBoxes` etc.)
    it can have arbitrary number of leading batch dimensions. For example,
    the image can have ``[..., C, H, W]`` shape. A bounding box can have ``[..., 4]`` shape.

    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or number, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x-axis in the range (-shear, +shear)
            will be applied. Else if shear is a sequence of 2 values a shear parallel to the x-axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a sequence of 4 values,
            an x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default.
        interpolation (InterpolationMode, optional): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        fill (number or tuple or dict, optional): Pixel fill value used when the  ``padding_mode`` is constant.
            Default is 0. If a tuple of length 3, it is used to fill R, G, B channels respectively.
            Fill value can be also a dictionary mapping data type to the fill value, e.g.
            ``fill={tv_tensors.Image: 127, tv_tensors.Mask: 0}`` where ``Image`` will be filled with 127 and
            ``Mask`` will be filled with 0.
        center (sequence, optional): Optional center of rotation, (x, y). Origin is the upper left corner.
            Default is the center of the image.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """
    _v1_transform_cls = _transforms.RandomAffine

    def __init__(self, degrees: Union[numbers.Number, Sequence], translate: Optional[Sequence[float]]=None, scale: Optional[Sequence[float]]=None, shear: Optional[Union[int, float, Sequence[float]]]=None, interpolation: Union[InterpolationMode, int]=InterpolationMode.NEAREST, fill: Union[_FillType, Dict[Union[Type, str], _FillType]]=0, center: Optional[List[float]]=None) -> None:
        super().__init__()
        self.degrees = _setup_angle(degrees, name='degrees', req_sizes=(2,))
        if translate is not None:
            _check_sequence_input(translate, 'translate', req_sizes=(2,))
            for t in translate:
                if not 0.0 <= t <= 1.0:
                    raise ValueError('translation values should be between 0 and 1')
        self.translate = translate
        if scale is not None:
            _check_sequence_input(scale, 'scale', req_sizes=(2,))
            for s in scale:
                if s <= 0:
                    raise ValueError('scale values should be positive')
        self.scale = scale
        if shear is not None:
            self.shear = _setup_angle(shear, name='shear', req_sizes=(2, 4))
        else:
            self.shear = shear
        self.interpolation = _check_interpolation(interpolation)
        self.fill = fill
        self._fill = _setup_fill_arg(fill)
        if center is not None:
            _check_sequence_input(center, 'center', req_sizes=(2,))
        self.center = center

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        height, width = query_size(flat_inputs)
        angle = torch.empty(1).uniform_(self.degrees[0], self.degrees[1]).item()
        if self.translate is not None:
            max_dx = float(self.translate[0] * width)
            max_dy = float(self.translate[1] * height)
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translate = (tx, ty)
        else:
            translate = (0, 0)
        if self.scale is not None:
            scale = torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
        else:
            scale = 1.0
        shear_x = shear_y = 0.0
        if self.shear is not None:
            shear_x = torch.empty(1).uniform_(self.shear[0], self.shear[1]).item()
            if len(self.shear) == 4:
                shear_y = torch.empty(1).uniform_(self.shear[2], self.shear[3]).item()
        shear = (shear_x, shear_y)
        return dict(angle=angle, translate=translate, scale=scale, shear=shear)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = _get_fill(self._fill, type(inpt))
        return self._call_kernel(F.affine, inpt, **params, interpolation=self.interpolation, fill=fill, center=self.center)