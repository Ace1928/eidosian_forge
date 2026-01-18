import collections.abc
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torchvision import transforms as _transforms
from torchvision.transforms.v2 import functional as F, Transform
from ._transform import _RandomApplyTransform
from ._utils import query_chw
class ColorJitter(Transform):
    """[BETA] Randomly change the brightness, contrast, saturation and hue of an image or video.

    .. v2betastatus:: ColorJitter transform

    If the input is a :class:`torch.Tensor`, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "I", "F" and modes with transparency (alpha channel) are not supported.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non-negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
            To jitter hue, the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
    """
    _v1_transform_cls = _transforms.ColorJitter

    def _extract_params_for_v1_transform(self) -> Dict[str, Any]:
        return {attr: value or 0 for attr, value in super()._extract_params_for_v1_transform().items()}

    def __init__(self, brightness: Optional[Union[float, Sequence[float]]]=None, contrast: Optional[Union[float, Sequence[float]]]=None, saturation: Optional[Union[float, Sequence[float]]]=None, hue: Optional[Union[float, Sequence[float]]]=None) -> None:
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    def _check_input(self, value: Optional[Union[float, Sequence[float]]], name: str, center: float=1.0, bound: Tuple[float, float]=(0, float('inf')), clip_first_on_zero: bool=True) -> Optional[Tuple[float, float]]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(f'If {name} is a single number, it must be non negative.')
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, collections.abc.Sequence) and len(value) == 2:
            value = [float(v) for v in value]
        else:
            raise TypeError(f'{name}={value} should be a single number or a sequence with length 2.')
        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError(f'{name} values should be between {bound}, but got {value}.')
        return None if value[0] == value[1] == center else (float(value[0]), float(value[1]))

    @staticmethod
    def _generate_value(left: float, right: float) -> float:
        return torch.empty(1).uniform_(left, right).item()

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        fn_idx = torch.randperm(4)
        b = None if self.brightness is None else self._generate_value(self.brightness[0], self.brightness[1])
        c = None if self.contrast is None else self._generate_value(self.contrast[0], self.contrast[1])
        s = None if self.saturation is None else self._generate_value(self.saturation[0], self.saturation[1])
        h = None if self.hue is None else self._generate_value(self.hue[0], self.hue[1])
        return dict(fn_idx=fn_idx, brightness_factor=b, contrast_factor=c, saturation_factor=s, hue_factor=h)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        output = inpt
        brightness_factor = params['brightness_factor']
        contrast_factor = params['contrast_factor']
        saturation_factor = params['saturation_factor']
        hue_factor = params['hue_factor']
        for fn_id in params['fn_idx']:
            if fn_id == 0 and brightness_factor is not None:
                output = self._call_kernel(F.adjust_brightness, output, brightness_factor=brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                output = self._call_kernel(F.adjust_contrast, output, contrast_factor=contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                output = self._call_kernel(F.adjust_saturation, output, saturation_factor=saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                output = self._call_kernel(F.adjust_hue, output, hue_factor=hue_factor)
        return output