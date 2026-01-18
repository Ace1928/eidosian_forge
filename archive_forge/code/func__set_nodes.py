from typing import List, Tuple
from ray.autoscaler._private.util import format_readonly_node_type
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def _set_nodes(self, nodes: List[Tuple[str, str]]):
    """Update the set of nodes in the cluster.

        Args:
            nodes: List of (node_id, node_manager_address) tuples.
        """
    new_nodes = {}
    for node_id, node_manager_address in nodes:
        new_nodes[node_id] = {'node_type': format_readonly_node_type(node_id), 'ip': node_manager_address}
    self.nodes = new_nodes