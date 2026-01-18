import collections
import copy
import functools
import itertools
import operator
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List
import torch
from transformers.file_utils import add_end_docstrings
from transformers.utils.fx import _gen_constructor_wrapper
def _merge_linears(self, graph_module: 'GraphModule', input_node: 'Node', linear_nodes: List['Node'], linears: List[torch.nn.Linear]):
    in_features = linears[0].in_features
    out_features = [linear.out_features for linear in linears]
    total_out_features = sum(out_features)
    use_bias = any((hasattr(linear, 'bias') for linear in linears))
    if use_bias and (not all((hasattr(linear, 'bias') for linear in linears))):
        warnings.warn('Not all the linear layers that are merged contain a bias, but some do. By merging, this is equivalent to adding a bias to the layers missing one.')
    merged_linear = torch.nn.Linear(in_features, total_out_features, bias=use_bias)
    dtype = linears[0].weight.dtype
    device = linears[0].weight.device
    with torch.no_grad():
        new_weight = torch.cat([linear.weight for linear in linears], dim=0).to(dtype=dtype, device=device)
        merged_linear.weight = torch.nn.Parameter(new_weight)
        if use_bias:
            new_bias = torch.cat([MergeLinears._get_bias(linear) for linear in linears], dim=0).to(dtype=dtype, device=device)
            merged_linear.bias = torch.nn.Parameter(new_bias)
    linear_module_names = [MergeLinears._get_linear_module_name(node) for node in linear_nodes]
    merged_linear_name = '-'.join(linear_module_names + ['merged'])
    fully_qualified_parent_name = linear_nodes[0].target.rsplit('.', maxsplit=1)[0]
    parent_module = graph_module.get_submodule(fully_qualified_parent_name)
    parent_module.add_module(merged_linear_name, merged_linear)
    for linear_node in linear_nodes:
        mod, name = MergeLinears._linear_node_to_module_and_attribute_name(graph_module, linear_node.target)
        delattr(mod, name)
    graph = graph_module.graph
    with graph.inserting_before(linear_nodes[0]):
        fully_qualified_merged_linear_name = '.'.join([fully_qualified_parent_name, merged_linear_name])
        merged_linear_node = graph.call_module(fully_qualified_merged_linear_name, args=(input_node,))
        self.mark_as_transformed(merged_linear_node)
        merged_linear_node.linear_node_targets = [n.target for n in linear_nodes]
    accum_out_features = list(itertools.accumulate([0] + out_features))
    for idx, node in enumerate(linear_nodes):
        node.op = 'call_function'
        node.target = operator.getitem
        slice_to_get = slice(accum_out_features[idx], accum_out_features[idx + 1])
        node.args = (merged_linear_node, (Ellipsis, slice_to_get))