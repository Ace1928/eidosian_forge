import functools
import warnings
from collections import OrderedDict
from inspect import getfullargspec, signature
from typing import Any, Callable, Dict, Optional, Tuple, Union
import torch
from torch.ao.quantization.quant_type import QuantType
from torch.fx import Node
from torch.nn.utils.parametrize import is_parametrized
def activation_is_int32_quantized(qconfig):
    """ Given a qconfig, decide if the activation needs to be
    quantized to int32 or not
    """
    return activation_dtype(qconfig) in [torch.qint32, torch.int32]