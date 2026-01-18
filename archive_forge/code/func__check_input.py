import collections.abc
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torchvision import transforms as _transforms
from torchvision.transforms.v2 import functional as F, Transform
from ._transform import _RandomApplyTransform
from ._utils import query_chw
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