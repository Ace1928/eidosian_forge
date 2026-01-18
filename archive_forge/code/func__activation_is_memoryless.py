from collections import namedtuple
from typing import Optional, Any, Union, Type
import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import (
from .observer import (
import warnings
import copy
def _activation_is_memoryless(qconfig: QConfig):
    """
    Return whether the observer for activations defined in the given QConfig is memoryless.
    This means a MovingAverage observer with averaging constant equal to 1.
    """

    def _is_memoryless(observer):
        return hasattr(observer, 'averaging_constant') and observer.averaging_constant == 1
    act = qconfig.activation()
    if isinstance(act, FakeQuantizeBase) and hasattr(act, 'activation_post_process'):
        return _is_memoryless(act.activation_post_process)
    else:
        return _is_memoryless(act)