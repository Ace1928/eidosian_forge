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
@register_annotator('max_pool2d')
def _annotate_max_pool2d(gm: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig], filter_fn: Optional[Callable[[Node], bool]]=None) -> Optional[List[List[Node]]]:
    module_partitions = get_source_partitions(gm.graph, [torch.nn.MaxPool2d, torch.nn.functional.max_pool2d], filter_fn)
    maxpool_partitions = list(itertools.chain(*module_partitions.values()))
    annotated_partitions = []
    for maxpool_partition in maxpool_partitions:
        annotated_partitions.append(maxpool_partition.nodes)
        output_node = maxpool_partition.output_nodes[0]
        maxpool_node = None
        for n in maxpool_partition.nodes:
            if n.target == torch.ops.aten.max_pool2d.default:
                maxpool_node = n
        assert maxpool_node is not None, 'XNNPACKQuantizer only works with torch.ops.aten.max_pool2d.default, '
        'please make sure you are exporting the model correctly'
        if _is_annotated([output_node, maxpool_node]):
            continue
        input_act = maxpool_node.args[0]
        assert isinstance(input_act, Node)
        if 'quantization_annotation' not in input_act.meta or not input_act.meta['quantization_annotation']._annotated or input_act.meta['quantization_annotation'].output_qspec is None:
            continue
        act_qspec = SharedQuantizationSpec(input_act)
        maxpool_node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={input_act: act_qspec}, _annotated=True)
        output_node.meta['quantization_annotation'] = QuantizationAnnotation(output_qspec=act_qspec, _annotated=True)
    return annotated_partitions