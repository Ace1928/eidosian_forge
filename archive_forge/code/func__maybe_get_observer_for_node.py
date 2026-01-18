from typing import Any, Dict, List, Optional, Set, Tuple, Union, Type, Callable
from torch.ao.quantization.quant_type import QuantType
import torch
import copy
import warnings
from torch.fx import (
from torch.fx.graph import (
from ..utils import (
from ..qconfig import (
from ..qconfig_mapping import QConfigMapping
from .qconfig_mapping_utils import (
from torch.ao.quantization.backend_config.utils import (
from torch.ao.quantization.backend_config import (
from torch.ao.quantization.observer import _is_activation_post_process
from .graph_module import (
from ._equalize import update_obs_for_equalization, convert_eq_obs
from torch.nn.utils.parametrize import type_before_parametrizations
from .utils import (
from torch.ao.quantization.utils import (
from torch.ao.quantization.quantize import (
from torch.ao.quantization.stubs import DeQuantStub
from .custom_config import (
from .lower_to_fbgemm import lower_to_fbgemm
from ._decomposed import quantized_decomposed_lib  # noqa: F401
import operator
def _maybe_get_observer_for_node(node: Node, modules: Dict[str, torch.nn.Module]) -> Optional[torch.nn.Module]:
    """
    If the node is observed, return the observer
    instance. Otherwise, return None.
    """
    for maybe_obs_node in node.users.keys():
        if maybe_obs_node.op == 'call_module':
            maybe_obs = modules[str(maybe_obs_node.target)]
            if _is_activation_post_process(maybe_obs):
                return maybe_obs
    return None