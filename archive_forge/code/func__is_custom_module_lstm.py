import copy
import torch
import torch.nn as nn
from torch.ao.quantization import (
from torch.ao.quantization.backend_config import (
from torch.ao.quantization.fake_quantize import (
from torch.ao.quantization.observer import (
from torch.ao.quantization.qconfig import (
from torch.ao.quantization.stubs import DeQuantStub
from torch.ao.quantization.utils import (
from torch.ao.quantization.observer import _is_activation_post_process
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.fx import GraphModule, map_arg
from torch.fx.graph import (
from .custom_config import PrepareCustomConfig
from ._decomposed import quantized_decomposed_lib  # noqa: F401
from typing import Callable, Optional, List, Dict, Any, Set, Tuple, Union, Type
from dataclasses import dataclass
from collections import namedtuple
import operator
import warnings
def _is_custom_module_lstm(node: Node, named_modules: Dict[str, torch.nn.Module], qconfig: QConfigAny=None, qhandler: Optional[Any]=None) -> bool:
    """
    Return whether this refers to the custom module LSTM flow.
    """
    mod = _get_module(node, named_modules)
    if qconfig is not None and qhandler is not None:
        assert isinstance(qhandler, torch.ao.quantization.fx.quantize_handler.QuantizeHandler)
        return isinstance(mod, torch.nn.LSTM) and activation_is_statically_quantized(qconfig) and qhandler.is_custom_module()
    else:
        return isinstance(mod, torch.ao.nn.quantizable.LSTM)