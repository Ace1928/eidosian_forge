import itertools
import operator
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional
import torch
import torch.nn.functional as F
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions
from torch.ao.quantization.pt2e.utils import (
from torch.ao.quantization.quantizer import (
from torch.ao.quantization.quantizer.utils import (
from torch.fx import Node
from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions
@register_annotator('gru_io_only')
def _annotate_gru_io_only(gm: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig], filter_fn: Optional[Callable[[Node], bool]]=None) -> Optional[List[List[Node]]]:
    gru_partitions = get_source_partitions(gm.graph, [torch.nn.GRU], filter_fn)
    gru_partitions = list(itertools.chain(*gru_partitions.values()))
    annotated_partitions = []
    for gru_partition in gru_partitions:
        annotated_partitions.append(gru_partition.nodes)
        output_nodes = gru_partition.output_nodes
        input_nodes = gru_partition.input_nodes
        if _is_annotated(input_nodes + output_nodes):
            continue
        input_qspec_map: Dict[Node, QuantizationSpecBase] = {}
        input_act = input_nodes[0]
        input_act_user = next(iter(input_act.users.keys()))
        assert isinstance(input_act, Node)
        assert isinstance(input_act_user, Node)
        input_act_user.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={input_act: get_input_act_qspec(quantization_config)}, _annotated=True)
        hidden_state = input_nodes[1]
        hidden_state_user = next(iter(hidden_state.users.keys()))
        assert isinstance(hidden_state, Node)
        assert isinstance(hidden_state_user, Node)
        hidden_state_user.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={hidden_state: get_input_act_qspec(quantization_config)}, _annotated=True)
        assert len(output_nodes) == 2, 'expecting GRU to have two outputs'
        for output in output_nodes:
            output.meta['quantization_annotation'] = QuantizationAnnotation(output_qspec=get_output_act_qspec(quantization_config), _annotated=True)
        nodes_to_mark_annotated = list(gru_partition.nodes)
        _mark_nodes_as_annotated(nodes_to_mark_annotated)
    return annotated_partitions