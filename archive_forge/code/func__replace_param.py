import functools
import logging
import math
import os
import warnings
from contextlib import ExitStack
from functools import partial
from types import ModuleType
from typing import Any, Callable, ContextManager, Literal, Optional, OrderedDict, Set, Tuple, Type, cast
import torch
from lightning_utilities import apply_to_collection
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.nn import init
from torch.nn.modules.module import _IncompatibleKeys
from typing_extensions import Self, override
from lightning_fabric.plugins.precision.precision import Precision
from lightning_fabric.plugins.precision.utils import (
from lightning_fabric.utilities.types import _DEVICE
def _replace_param(param: torch.nn.Parameter, data: torch.Tensor, quant_state: Optional[Tuple]=None) -> torch.nn.Parameter:
    bnb = _import_bitsandbytes()
    if param.device.type == 'meta':
        if isinstance(param, bnb.nn.Params4bit):
            return bnb.nn.Params4bit(data, requires_grad=data.requires_grad, quant_state=quant_state, compress_statistics=param.compress_statistics, quant_type=param.quant_type)
        return torch.nn.Parameter(data, requires_grad=data.requires_grad)
    param.data = data
    if isinstance(param, bnb.nn.Params4bit):
        param.quant_state = quant_state
    return param