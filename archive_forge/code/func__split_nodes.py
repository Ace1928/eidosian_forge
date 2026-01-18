import logging
from typing import Dict, List, Set
import torch
import torch.fx
from torch.fx.node import Node
def _split_nodes(traced_graph_module: torch.fx.GraphModule, shard_count: int=3) -> Dict:
    """Utility used to trace a graph and identify shard cutpoints."""
    node_name_to_shard_id: Dict[str, int] = {}
    shard_id = 0
    nodes_so_far = []
    param_count: Dict[str, int] = {}
    shard_to_param_count = {}
    for name, module in traced_graph_module.named_modules():
        name = name.replace('.', '_')
        param_count[name] = sum([x.numel() for x in module.parameters()])
    logging.info(f'Total number of params are {param_count['']}')
    per_shard_param = param_count[''] // shard_count
    logging.info(f'Per shard param count {per_shard_param}')
    for node in traced_graph_module.graph.nodes:
        if node.op == 'placeholder':
            node_name_to_shard_id[node.name] = shard_id
            nodes_so_far.append(node.name)
        elif node.op in ['get_attr', 'call_function', 'call_method', 'call_module']:
            min_shard_id = shard_id
            min_node_name = ''
            for arg in node.args:
                if not hasattr(arg, 'name'):
                    continue
                if arg.name in node_name_to_shard_id and arg.name != nodes_so_far[-1]:
                    if node_name_to_shard_id[arg.name] < min_shard_id:
                        min_shard_id = node_name_to_shard_id[arg.name]
                        min_node_name = arg.name
            if min_shard_id < shard_id:
                for node_name in reversed(nodes_so_far):
                    node_name_to_shard_id[node_name] = min_shard_id
                    if node_name == min_node_name:
                        break
                shard_id = min_shard_id
                shard_to_param_count = _create_shard_to_param_count(param_count, node_name_to_shard_id)
            node_name_to_shard_id[node.name] = shard_id
            nodes_so_far.append(node.name)
            shard_to_param_count = _create_shard_to_param_count(param_count, node_name_to_shard_id)
            if shard_id in shard_to_param_count and shard_to_param_count[shard_id] > per_shard_param:
                shard_id += 1
        elif node.op == 'output':
            break
    return node_name_to_shard_id