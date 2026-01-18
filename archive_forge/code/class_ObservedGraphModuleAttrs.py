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
@dataclass
class ObservedGraphModuleAttrs:
    node_name_to_qconfig: Dict[str, QConfigAny]
    node_name_to_scope: Dict[str, Tuple[str, type]]
    prepare_custom_config: PrepareCustomConfig
    equalization_node_name_to_qconfig: Dict[str, Any]
    qconfig_mapping: QConfigMapping
    is_qat: bool
    observed_node_names: Set[str]
    is_observed_standalone_module: bool = False
    standalone_module_input_quantized_idxs: Optional[List[int]] = None
    standalone_module_output_quantized_idxs: Optional[List[int]] = None