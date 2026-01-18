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
@register_annotator('adaptive_avg_pool2d')
def _annotate_adaptive_avg_pool2d(gm: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig], filter_fn: Optional[Callable[[Node], bool]]=None) -> Optional[List[List[Node]]]:
    """Always annotate adaptive_avg_pool2d op"""
    module_partitions = get_source_partitions(gm.graph, [torch.nn.AdaptiveAvgPool2d, F.adaptive_avg_pool2d], filter_fn)
    partitions = list(itertools.chain(*module_partitions.values()))
    annotated_partitions = []
    for partition in partitions:
        pool_node = partition.output_nodes[0]
        if pool_node.op != 'call_function' or pool_node.target != torch.ops.aten.adaptive_avg_pool2d.default:
            raise ValueError(f'{pool_node} is not an aten adaptive_avg_pool2d operator')
        if _is_annotated([pool_node]):
            continue
        annotated_partitions.append(partition.nodes)
        input_act = pool_node.args[0]
        assert isinstance(input_act, Node)
        if 'quantization_annotation' not in input_act.meta or not input_act.meta['quantization_annotation']._annotated or input_act.meta['quantization_annotation'].output_qspec is None:
            input_act_qspec = get_input_act_qspec(quantization_config)
        else:
            input_act_qspec = SharedQuantizationSpec(input_act)
        output_act_qspec = SharedQuantizationSpec((input_act, pool_node))
        pool_node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={input_act: input_act_qspec}, output_qspec=output_act_qspec, _annotated=True)
    return annotated_partitions