import collections.abc
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torchvision import transforms as _transforms
from torchvision.transforms.v2 import functional as F, Transform
from ._transform import _RandomApplyTransform
from ._utils import query_chw
class RandomPhotometricDistort(Transform):
    """[BETA] Randomly distorts the image or video as used in `SSD: Single Shot
    MultiBox Detector <https://arxiv.org/abs/1512.02325>`_.

    .. v2betastatus:: RandomPhotometricDistort transform

    This transform relies on :class:`~torchvision.transforms.v2.ColorJitter`
    under the hood to adjust the contrast, saturation, hue, brightness, and also
    randomly permutes channels.

    Args:
        brightness (tuple of float (min, max), optional): How much to jitter brightness.
            brightness_factor is chosen uniformly from [min, max]. Should be non negative numbers.
        contrast tuple of float (min, max), optional): How much to jitter contrast.
            contrast_factor is chosen uniformly from [min, max]. Should be non-negative numbers.
        saturation (tuple of float (min, max), optional): How much to jitter saturation.
            saturation_factor is chosen uniformly from [min, max]. Should be non negative numbers.
        hue (tuple of float (min, max), optional): How much to jitter hue.
            hue_factor is chosen uniformly from [min, max].  Should have -0.5 <= min <= max <= 0.5.
            To jitter hue, the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
        p (float, optional) probability each distortion operation (contrast, saturation, ...) to be applied.
            Default is 0.5.
    """

    def __init__(self, brightness: Tuple[float, float]=(0.875, 1.125), contrast: Tuple[float, float]=(0.5, 1.5), saturation: Tuple[float, float]=(0.5, 1.5), hue: Tuple[float, float]=(-0.05, 0.05), p: float=0.5):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.hue = hue
        self.saturation = saturation
        self.p = p

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        num_channels, *_ = query_chw(flat_inputs)
        params: Dict[str, Any] = {key: ColorJitter._generate_value(range[0], range[1]) if torch.rand(1) < self.p else None for key, range in [('brightness_factor', self.brightness), ('contrast_factor', self.contrast), ('saturation_factor', self.saturation), ('hue_factor', self.hue)]}
        params['contrast_before'] = bool(torch.rand(()) < 0.5)
        params['channel_permutation'] = torch.randperm(num_channels) if torch.rand(1) < self.p else None
        return params

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if params['brightness_factor'] is not None:
            inpt = self._call_kernel(F.adjust_brightness, inpt, brightness_factor=params['brightness_factor'])
        if params['contrast_factor'] is not None and params['contrast_before']:
            inpt = self._call_kernel(F.adjust_contrast, inpt, contrast_factor=params['contrast_factor'])
        if params['saturation_factor'] is not None:
            inpt = self._call_kernel(F.adjust_saturation, inpt, saturation_factor=params['saturation_factor'])
        if params['hue_factor'] is not None:
            inpt = self._call_kernel(F.adjust_hue, inpt, hue_factor=params['hue_factor'])
        if params['contrast_factor'] is not None and (not params['contrast_before']):
            inpt = self._call_kernel(F.adjust_contrast, inpt, contrast_factor=params['contrast_factor'])
        if params['channel_permutation'] is not None:
            inpt = self._call_kernel(F.permute_channels, inpt, permutation=params['channel_permutation'])
        return inpt