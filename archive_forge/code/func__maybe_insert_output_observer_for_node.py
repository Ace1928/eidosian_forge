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
def _maybe_insert_output_observer_for_node(node: Node, model: torch.nn.Module, named_modules: Dict[str, torch.nn.Module], graph: Graph, obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize], is_qat: bool) -> Optional[Node]:
    if node in obs_or_fq_map:
        output_act_obs_or_fq = obs_or_fq_map[node]
        return _insert_obs_or_fq(node, output_act_obs_or_fq, model, named_modules, graph)
    return None