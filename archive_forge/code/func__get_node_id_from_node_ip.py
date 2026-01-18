from typing import Dict, Optional, Union
import ray
def _get_node_id_from_node_ip(node_ip: str) -> Optional[str]:
    """Returns the node ID for the first alive node with the input IP."""
    for node in ray.nodes():
        if node['Alive'] and node['NodeManagerAddress'] == node_ip:
            return node['NodeID']
    return None