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
class RandomResize(Transform):
    """[BETA] Randomly resize the input.

    .. v2betastatus:: RandomResize transform

    This transformation can be used together with ``RandomCrop`` as data augmentations to train
    models on image segmentation task.

    Output spatial size is randomly sampled from the interval ``[min_size, max_size]``:

    .. code-block:: python

        size = uniform_sample(min_size, max_size)
        output_width = size
        output_height = size

    If the input is a :class:`torch.Tensor` or a ``TVTensor`` (e.g. :class:`~torchvision.tv_tensors.Image`,
    :class:`~torchvision.tv_tensors.Video`, :class:`~torchvision.tv_tensors.BoundingBoxes` etc.)
    it can have arbitrary number of leading batch dimensions. For example,
    the image can have ``[..., C, H, W]`` shape. A bounding box can have ``[..., 4]`` shape.

    Args:
        min_size (int): Minimum output size for random sampling
        max_size (int): Maximum output size for random sampling
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

    def __init__(self, min_size: int, max_size: int, interpolation: Union[InterpolationMode, int]=InterpolationMode.BILINEAR, antialias: Optional[Union[str, bool]]='warn') -> None:
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = _check_interpolation(interpolation)
        self.antialias = antialias

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        size = int(torch.randint(self.min_size, self.max_size, ()))
        return dict(size=[size])

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.resize, inpt, params['size'], interpolation=self.interpolation, antialias=self.antialias)