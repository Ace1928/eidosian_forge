import dataclasses
import itertools
import operator
from typing import Any, Callable, Dict, List, Tuple
import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.subgraph_rewriter import (
import torch.nn.functional as F
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.ao.quantization.quantizer import (
from .utils import (
def _update_special_qspecs_after_replacement(node: Node, original_to_replacement_node: Dict[Node, Node]):
    """
    Update the `SharedQuantizationSpec`s and `DerivedQuantizationSpec`s
    used in `node`'s quantization annotation after subgraph rewriting.

    The original annotation referred to the nodes in the original graph,
    so the nodes used in these special quantization specs will need to
    be updated to the corresponding nodes in the replacement graph.
    """

    def _get_new_edge_or_node(edge_or_node: EdgeOrNode):
        if isinstance(edge_or_node, Node):
            _node = edge_or_node
            return original_to_replacement_node.get(_node, _node)
        elif isinstance(edge_or_node, tuple) and len(edge_or_node) == 2 and all((isinstance(x, Node) for x in edge_or_node)):
            src, dest = edge_or_node
            return (original_to_replacement_node.get(src, src), original_to_replacement_node.get(dest, dest))
        else:
            raise ValueError('unexpected type for edge_or_node: ', type(edge_or_node))

    def _get_new_qspec(qspec: QuantizationSpecBase):
        if isinstance(qspec, SharedQuantizationSpec):
            new_edge_or_node = _get_new_edge_or_node(qspec.edge_or_node)
            return SharedQuantizationSpec(new_edge_or_node)
        elif isinstance(qspec, DerivedQuantizationSpec):
            new_derived_from = [_get_new_edge_or_node(x) for x in qspec.derived_from]
            return dataclasses.replace(qspec, derived_from=new_derived_from)
        else:
            return qspec
    if 'quantization_annotation' not in node.meta:
        return
    annotation = node.meta['quantization_annotation']
    for input_node, qspec in annotation.input_qspec_map.items():
        annotation.input_qspec_map[input_node] = _get_new_qspec(qspec)
    annotation.output_qspec = _get_new_qspec(annotation.output_qspec)