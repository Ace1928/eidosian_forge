from collections import namedtuple
from typing import Optional, Any, Union, Type
import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import (
from .observer import (
import warnings
import copy
def _is_memoryless(observer):
    return hasattr(observer, 'averaging_constant') and observer.averaging_constant == 1