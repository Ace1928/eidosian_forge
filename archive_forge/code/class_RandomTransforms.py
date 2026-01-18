import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from ..utils import _log_api_usage_once
from . import functional as F
from .functional import _interpolation_modes_from_int, InterpolationMode
class RandomTransforms:
    """Base class for a list of transformations with randomness

    Args:
        transforms (sequence): list of transformations
    """

    def __init__(self, transforms):
        _log_api_usage_once(self)
        if not isinstance(transforms, Sequence):
            raise TypeError('Argument transforms should be a sequence')
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string