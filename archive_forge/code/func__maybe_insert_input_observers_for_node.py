import torch
from torch._subclasses import FakeTensor
from torch.ao.quantization.fx.prepare import (
from torch.fx import (
from torch.fx.node import Argument
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
from typing import Dict, Tuple, Union, Any, Optional
from torch.ao.quantization.quantizer import (
from torch.ao.quantization import ObserverOrFakeQuantize
def _maybe_insert_input_observers_for_node(node: Node, qconfig: QConfigAny, model: torch.nn.Module, named_modules: Dict[str, torch.nn.Module], obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize], is_qat: bool) -> None:
    """
    If needed, inserts observers to the input args and kwargs of `node`.
    Note: modifies `node` inplace.

    For example, if cur_node needs an observer after prev_node, we change from

      prev_node -> cur_node

    To

      prev_node -> obs -> cur_node

    """
    new_args = []
    for arg in node.args:
        new_arg = _maybe_insert_input_observer_for_arg_or_kwarg(node, arg, qconfig, model, named_modules, obs_or_fq_map, is_qat)
        new_args.append(new_arg)
    assert node.target == torch.ops.aten.clone.default or node.target == torch.ops.aten.zeros_like.default or len(node.kwargs) == 0, ' expecting kwargs for aten op IR to be empty'
    node.args = tuple(new_args)