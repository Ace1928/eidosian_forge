import collections
import torch
import torch.nn as nn
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.ao.ns.fx.mappings import (
from torch.ao.ns.fx.graph_matcher import (
from .fx.weight_utils import (
from .fx.graph_passes import (
from .fx.utils import (
from .fx.ns_types import (
from torch.ao.quantization.backend_config.utils import get_fusion_pattern_to_root_node_getter
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.fx.match_utils import _find_matches
from torch.ao.quantization.fx.graph_module import _get_observed_graph_module_attr
from torch.ao.quantization.fx.qconfig_mapping_utils import _generate_node_name_to_qconfig
from torch.ao.quantization.fx.quantize_handler import _get_pattern_to_quantize_handlers
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization import QConfigMapping
from torch.ao.ns.fx.n_shadows_utils import (
from torch.ao.ns.fx.qconfig_multi_mapping import QConfigMultiMapping
from typing import Dict, Tuple, Callable, List, Optional, Set, Any, Type
class OutputLogger(nn.Module):
    """
    Base class for capturing intermediate values.
    """
    stats: List[torch.Tensor]
    stats_rnn: List[RNNReturnType]
    _is_impure = True

    def __init__(self, ref_node_name: str, prev_node_name: str, model_name: str, ref_name: str, prev_node_target_type: str, ref_node_target_type: str, results_type: str, index_within_arg: int, index_of_arg: int, fqn: Optional[str], qconfig_str: Optional[str]=''):
        super().__init__()
        self.stats: List[torch.Tensor] = []
        self.stats_rnn: List[RNNReturnType] = []
        self.ref_node_name = ref_node_name
        self.prev_node_name = prev_node_name
        self.model_name = model_name
        self.ref_name = ref_name
        self.prev_node_target_type = prev_node_target_type
        self.ref_node_target_type = ref_node_target_type
        self.results_type = results_type
        self.index_within_arg = index_within_arg
        self.index_of_arg = index_of_arg
        self.fqn = fqn
        self.enabled = True
        self.qconfig_str = qconfig_str
        self.save_activations = True

    def forward(self, x):
        """
        """
        if not self.enabled:
            return x
        if not self.save_activations:
            return x
        if isinstance(x, torch.Tensor):
            self.stats.append(x.detach())
        elif isinstance(x, tuple) and len(x) == 2 and (len(x[1]) == 2):
            new_res = (x[0].detach(), (x[1][0].detach(), x[1][1].detach()))
            self.stats_rnn.append(new_res)
        return x

    def __repr__(self):
        clean_dict = {k: v for k, v in self.__dict__.items() if k != 'training' and (not k.startswith('_'))}
        return f'OutputLogger({clean_dict})'