import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import PIL.Image
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec
from torchvision import transforms as _transforms, tv_tensors
from torchvision.transforms import _functional_tensor as _FT
from torchvision.transforms.v2 import AutoAugmentPolicy, functional as F, InterpolationMode, Transform
from torchvision.transforms.v2.functional._geometry import _check_interpolation
from torchvision.transforms.v2.functional._meta import get_size
from torchvision.transforms.v2.functional._utils import _FillType, _FillTypeJIT
from ._utils import _get_fill, _setup_fill_arg, check_type, is_pure_tensor
class RandAugment(_AutoAugmentBase):
    """[BETA] RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.

    .. v2betastatus:: RandAugment transform

    This transformation works on images and videos only.

    If the input is :class:`torch.Tensor`, it should be of type ``torch.uint8``, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int, optional): Number of augmentation transformations to apply sequentially.
        magnitude (int, optional): Magnitude for all the transformations.
        num_magnitude_bins (int, optional): The number of different magnitude values.
        interpolation (InterpolationMode, optional): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """
    _v1_transform_cls = _transforms.RandAugment
    _AUGMENTATION_SPACE = {'Identity': (lambda num_bins, height, width: None, False), 'ShearX': (lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins), True), 'ShearY': (lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins), True), 'TranslateX': (lambda num_bins, height, width: torch.linspace(0.0, 150.0 / 331.0 * width, num_bins), True), 'TranslateY': (lambda num_bins, height, width: torch.linspace(0.0, 150.0 / 331.0 * height, num_bins), True), 'Rotate': (lambda num_bins, height, width: torch.linspace(0.0, 30.0, num_bins), True), 'Brightness': (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True), 'Color': (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True), 'Contrast': (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True), 'Sharpness': (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True), 'Posterize': (lambda num_bins, height, width: (8 - torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False), 'Solarize': (lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins), False), 'AutoContrast': (lambda num_bins, height, width: None, False), 'Equalize': (lambda num_bins, height, width: None, False)}

    def __init__(self, num_ops: int=2, magnitude: int=9, num_magnitude_bins: int=31, interpolation: Union[InterpolationMode, int]=InterpolationMode.NEAREST, fill: Union[_FillType, Dict[Union[Type, str], _FillType]]=None) -> None:
        super().__init__(interpolation=interpolation, fill=fill)
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins

    def forward(self, *inputs: Any) -> Any:
        flat_inputs_with_spec, image_or_video = self._flatten_and_extract_image_or_video(inputs)
        height, width = get_size(image_or_video)
        for _ in range(self.num_ops):
            transform_id, (magnitudes_fn, signed) = self._get_random_item(self._AUGMENTATION_SPACE)
            magnitudes = magnitudes_fn(self.num_magnitude_bins, height, width)
            if magnitudes is not None:
                magnitude = float(magnitudes[self.magnitude])
                if signed and torch.rand(()) <= 0.5:
                    magnitude *= -1
            else:
                magnitude = 0.0
            image_or_video = self._apply_image_or_video_transform(image_or_video, transform_id, magnitude, interpolation=self.interpolation, fill=self._fill)
        return self._unflatten_and_insert_image_or_video(flat_inputs_with_spec, image_or_video)