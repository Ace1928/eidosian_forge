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
class ElasticTransform(Transform):
    """[BETA] Transform the input with elastic transformations.

    .. v2betastatus:: RandomPerspective transform

    If the input is a :class:`torch.Tensor` or a ``TVTensor`` (e.g. :class:`~torchvision.tv_tensors.Image`,
    :class:`~torchvision.tv_tensors.Video`, :class:`~torchvision.tv_tensors.BoundingBoxes` etc.)
    it can have arbitrary number of leading batch dimensions. For example,
    the image can have ``[..., C, H, W]`` shape. A bounding box can have ``[..., 4]`` shape.

    Given alpha and sigma, it will generate displacement
    vectors for all pixels based on random offsets. Alpha controls the strength
    and sigma controls the smoothness of the displacements.
    The displacements are added to an identity grid and the resulting grid is
    used to transform the input.

    .. note::
        Implementation to transform bounding boxes is approximative (not exact).
        We construct an approximation of the inverse grid as ``inverse_grid = identity - displacement``.
        This is not an exact inverse of the grid used to transform images, i.e. ``grid = identity + displacement``.
        Our assumption is that ``displacement * displacement`` is small and can be ignored.
        Large displacements would lead to large errors in the approximation.

    Applications:
        Randomly transforms the morphology of objects in images and produces a
        see-through-water-like effect.

    Args:
        alpha (float or sequence of floats, optional): Magnitude of displacements. Default is 50.0.
        sigma (float or sequence of floats, optional): Smoothness of displacements. Default is 5.0.
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
    _v1_transform_cls = _transforms.ElasticTransform

    def __init__(self, alpha: Union[float, Sequence[float]]=50.0, sigma: Union[float, Sequence[float]]=5.0, interpolation: Union[InterpolationMode, int]=InterpolationMode.BILINEAR, fill: Union[_FillType, Dict[Union[Type, str], _FillType]]=0) -> None:
        super().__init__()
        self.alpha = _setup_number_or_seq(alpha, 'alpha')
        self.sigma = _setup_number_or_seq(sigma, 'sigma')
        self.interpolation = _check_interpolation(interpolation)
        self.fill = fill
        self._fill = _setup_fill_arg(fill)

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        size = list(query_size(flat_inputs))
        dx = torch.rand([1, 1] + size) * 2 - 1
        if self.sigma[0] > 0.0:
            kx = int(8 * self.sigma[0] + 1)
            if kx % 2 == 0:
                kx += 1
            dx = self._call_kernel(F.gaussian_blur, dx, [kx, kx], list(self.sigma))
        dx = dx * self.alpha[0] / size[0]
        dy = torch.rand([1, 1] + size) * 2 - 1
        if self.sigma[1] > 0.0:
            ky = int(8 * self.sigma[1] + 1)
            if ky % 2 == 0:
                ky += 1
            dy = self._call_kernel(F.gaussian_blur, dy, [ky, ky], list(self.sigma))
        dy = dy * self.alpha[1] / size[1]
        displacement = torch.concat([dx, dy], 1).permute([0, 2, 3, 1])
        return dict(displacement=displacement)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = _get_fill(self._fill, type(inpt))
        return self._call_kernel(F.elastic, inpt, **params, fill=fill, interpolation=self.interpolation)