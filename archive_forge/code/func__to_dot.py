import hashlib
import torch
import torch.fx
from typing import Any, Dict, Optional, TYPE_CHECKING
from torch.fx.node import _get_qualified_name, _format_arg
from torch.fx.graph import _parse_stack_trace
from torch.fx.passes.shape_prop import TensorMetadata
from torch.fx._compatibility import compatibility
from itertools import chain
def _to_dot(self, graph_module: torch.fx.GraphModule, name: str, ignore_getattr: bool, ignore_parameters_and_buffers: bool, skip_node_names_in_args: bool, parse_stack_trace: bool) -> pydot.Dot:
    """
            Actual interface to visualize a fx.Graph. Note that it takes in the GraphModule instead of the Graph.
            If ignore_parameters_and_buffers is True, the parameters and buffers
            created with the module will not be added as nodes and edges.
            """
    dot_graph = pydot.Dot(name, rankdir='TB')
    buf_name_to_subgraph = {}
    for node in graph_module.graph.nodes:
        if ignore_getattr and node.op == 'get_attr':
            continue
        style = self._get_node_style(node)
        dot_node = pydot.Node(node.name, label=self._get_node_label(graph_module, node, skip_node_names_in_args, parse_stack_trace), **style)
        current_graph = dot_graph
        buf_meta = node.meta.get('buf_meta', None)
        if buf_meta is not None and buf_meta.n_origin > 1:
            buf_name = buf_meta.name
            if buf_name not in buf_name_to_subgraph:
                buf_name_to_subgraph[buf_name] = pydot.Cluster(buf_name, label=buf_name)
            current_graph = buf_name_to_subgraph.get(buf_name)
        current_graph.add_node(dot_node)

        def get_module_params_or_buffers():
            for pname, ptensor in chain(leaf_module.named_parameters(), leaf_module.named_buffers()):
                pname1 = node.name + '.' + pname
                label1 = pname1 + '|op_code=get_' + 'parameter' if isinstance(ptensor, torch.nn.Parameter) else 'buffer' + '\\l'
                dot_w_node = pydot.Node(pname1, label='{' + label1 + self._get_tensor_label(ptensor) + '}', **_WEIGHT_TEMPLATE)
                dot_graph.add_node(dot_w_node)
                dot_graph.add_edge(pydot.Edge(pname1, node.name))
        if node.op == 'call_module':
            leaf_module = self._get_leaf_node(graph_module, node)
            if not ignore_parameters_and_buffers and (not isinstance(leaf_module, torch.fx.GraphModule)):
                get_module_params_or_buffers()
    for subgraph in buf_name_to_subgraph.values():
        subgraph.set('color', 'royalblue')
        subgraph.set('penwidth', '2')
        dot_graph.add_subgraph(subgraph)
    for node in graph_module.graph.nodes:
        if ignore_getattr and node.op == 'get_attr':
            continue
        for user in node.users:
            dot_graph.add_edge(pydot.Edge(node.name, user.name))
    return dot_graph