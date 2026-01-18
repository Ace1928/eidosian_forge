import logging
from typing import Dict, List, Set
import torch
import torch.fx
from torch.fx.node import Node
def _get_count(param_count: Dict, node_name: str) -> int:
    """Identify different mutations of a given node name."""
    if node_name in param_count:
        return param_count[node_name]
    elif node_name.split('_')[0] in param_count:
        return param_count[node_name.split('_')[0]]
    else:
        raise RuntimeError(f'Unable to find match between param {param_count} and node {node_name}')