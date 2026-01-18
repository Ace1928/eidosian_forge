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
def _get_observer_from_activation_post_process(activation_post_process: Union[ObserverBase, FakeQuantizeBase]) -> ObserverBase:
    """
    If `activation_post_process` is an observer, return the observer.
    If `activation_post_process` is a fake quantize, return the internal observer.
    """
    if isinstance(activation_post_process, ObserverBase):
        return activation_post_process
    else:
        assert isinstance(activation_post_process, FakeQuantizeBase)
        return activation_post_process.activation_post_process