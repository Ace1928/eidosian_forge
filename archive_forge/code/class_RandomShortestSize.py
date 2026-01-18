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
class RandomShortestSize(Transform):
    """[BETA] Randomly resize the input.

    .. v2betastatus:: RandomShortestSize transform

    If the input is a :class:`torch.Tensor` or a ``TVTensor`` (e.g. :class:`~torchvision.tv_tensors.Image`,
    :class:`~torchvision.tv_tensors.Video`, :class:`~torchvision.tv_tensors.BoundingBoxes` etc.)
    it can have arbitrary number of leading batch dimensions. For example,
    the image can have ``[..., C, H, W]`` shape. A bounding box can have ``[..., 4]`` shape.

    Args:
        min_size (int or sequence of int): Minimum spatial size. Single integer value or a sequence of integer values.
        max_size (int, optional): Maximum spatial size. Default, None.
        interpolation (InterpolationMode, optional): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.NEAREST_EXACT``,
            ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        antialias (bool, optional): Whether to apply antialiasing.
            It only affects **tensors** with bilinear or bicubic modes and it is
            ignored otherwise: on PIL images, antialiasing is always applied on
            bilinear or bicubic modes; on other modes (for PIL images and
            tensors), antialiasing makes no sense and this parameter is ignored.
            Possible values are:

            - ``True``: will apply antialiasing for bilinear or bicubic modes.
              Other mode aren't affected. This is probably what you want to use.
            - ``False``: will not apply antialiasing for tensors on any mode. PIL
              images are still antialiased on bilinear or bicubic modes, because
              PIL doesn't support no antialias.
            - ``None``: equivalent to ``False`` for tensors and ``True`` for
              PIL images. This value exists for legacy reasons and you probably
              don't want to use it unless you really know what you are doing.

            The current default is ``None`` **but will change to** ``True`` **in
            v0.17** for the PIL and Tensor backends to be consistent.
    """

    def __init__(self, min_size: Union[List[int], Tuple[int], int], max_size: Optional[int]=None, interpolation: Union[InterpolationMode, int]=InterpolationMode.BILINEAR, antialias: Optional[Union[str, bool]]='warn'):
        super().__init__()
        self.min_size = [min_size] if isinstance(min_size, int) else list(min_size)
        self.max_size = max_size
        self.interpolation = _check_interpolation(interpolation)
        self.antialias = antialias

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        orig_height, orig_width = query_size(flat_inputs)
        min_size = self.min_size[int(torch.randint(len(self.min_size), ()))]
        r = min_size / min(orig_height, orig_width)
        if self.max_size is not None:
            r = min(r, self.max_size / max(orig_height, orig_width))
        new_width = int(orig_width * r)
        new_height = int(orig_height * r)
        return dict(size=(new_height, new_width))

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.resize, inpt, size=params['size'], interpolation=self.interpolation, antialias=self.antialias)