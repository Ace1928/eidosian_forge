import functools
import warnings
from collections import OrderedDict
from inspect import getfullargspec, signature
from typing import Any, Callable, Dict, Optional, Tuple, Union
import torch
from torch.ao.quantization.quant_type import QuantType
from torch.fx import Node
from torch.nn.utils.parametrize import is_parametrized
def activation_is_statically_quantized(qconfig):
    """ Given a qconfig, decide if the activation needs to be
    quantized or not, this includes quantizing to quint8, qint8 and qint32 and float16
    """
    return activation_dtype(qconfig) in [torch.quint8, torch.qint8, torch.qint32, torch.float16, torch.uint8, torch.int8, torch.int16, torch.int32] and (not activation_is_dynamically_quantized(qconfig))