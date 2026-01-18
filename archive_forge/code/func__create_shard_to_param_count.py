import logging
from typing import Dict, List, Set
import torch
import torch.fx
from torch.fx.node import Node
def _create_shard_to_param_count(param_count: Dict, node_name_to_shard_id: Dict) -> Dict:
    """Utility to create a map from shard id to param count using existing state."""
    shard_to_param_count: Dict[int, int] = {}
    for node_name in node_name_to_shard_id.keys():
        try:
            count = _get_count(param_count, node_name)
        except RuntimeError:
            continue
        if node_name_to_shard_id[node_name] in shard_to_param_count:
            shard_to_param_count[node_name_to_shard_id[node_name]] += count
        else:
            shard_to_param_count[node_name_to_shard_id[node_name]] = count
    return shard_to_param_count