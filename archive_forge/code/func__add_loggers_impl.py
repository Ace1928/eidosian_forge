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
def _add_loggers_impl(name_a: str, gm_a: GraphModule, name_b: str, gm_b: GraphModule, logger_cls: Callable, should_log_inputs: bool, base_name_to_sets_of_related_ops: Optional[Dict[str, Set[NSNodeTargetType]]]=None, unmatchable_types_map: Optional[Dict[str, Set[NSNodeTargetType]]]=None) -> Tuple[nn.Module, nn.Module]:
    torch._C._log_api_usage_once('quantization_api._numeric_suite_fx._add_loggers_impl')
    matched_subgraph_pairs = get_matching_subgraph_pairs(gm_a, gm_b, base_name_to_sets_of_related_ops, unmatchable_types_map)
    nodes_and_names_to_instrument_inputs_a = []
    nodes_and_names_to_instrument_inputs_b = []
    nodes_and_names_to_instrument_outputs_a = []
    nodes_and_names_to_instrument_outputs_b = []
    for match_name, (subgraph_a, subgraph_b) in matched_subgraph_pairs.items():
        ref_node_type_a = get_target_type_str(subgraph_a.base_op_node, gm_a)
        ref_node_type_b = get_target_type_str(subgraph_b.base_op_node, gm_b)
        if should_log_inputs:
            nodes_and_names_to_instrument_inputs_a.append((subgraph_a.start_node, match_name, ref_node_type_a))
            nodes_and_names_to_instrument_inputs_b.append((subgraph_b.start_node, match_name, ref_node_type_b))
        nodes_and_names_to_instrument_outputs_a.append((subgraph_a.end_node, match_name, ref_node_type_a))
        nodes_and_names_to_instrument_outputs_b.append((subgraph_b.end_node, match_name, ref_node_type_b))
    new_model_a = _add_loggers_one_model(name_a, gm_a, nodes_and_names_to_instrument_inputs_a, nodes_and_names_to_instrument_outputs_a, logger_cls)
    new_model_b = _add_loggers_one_model(name_b, gm_b, nodes_and_names_to_instrument_inputs_b, nodes_and_names_to_instrument_outputs_b, logger_cls)
    return (new_model_a, new_model_b)