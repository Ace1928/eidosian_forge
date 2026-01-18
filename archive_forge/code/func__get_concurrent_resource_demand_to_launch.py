import collections
import copy
import logging
import os
from abc import abstractmethod
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple
import ray
from ray._private.gcs_utils import PlacementGroupTableData
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.loader import load_function_or_class
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.util import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
from ray.core.generated.common_pb2 import PlacementStrategy
def _get_concurrent_resource_demand_to_launch(self, to_launch: Dict[NodeType, int], connected_nodes: List[NodeIP], non_terminated_nodes: List[NodeID], pending_launches_nodes: Dict[NodeType, int], adjusted_min_workers: Dict[NodeType, int], placement_group_nodes: Dict[NodeType, int]) -> Dict[NodeType, int]:
    """Updates the max concurrent resources to launch for each node type.

        Given the current nodes that should be launched, the non terminated
        nodes (running and pending) and the pending to be launched nodes. This
        method calculates the maximum number of nodes to launch concurrently
        for each node type as follows:
            1) Calculates the running nodes.
            2) Calculates the pending nodes and gets the launching nodes.
            3) Limits the total number of pending + currently-launching +
               to-be-launched nodes to:
                   max(
                       5,
                       self.upscaling_speed * max(running_nodes[node_type], 1)
                   ).

        Args:
            to_launch: List of number of nodes to launch based on resource
                demand for every node type.
            connected_nodes: Running nodes (from LoadMetrics).
            non_terminated_nodes: Non terminated nodes (pending/running).
            pending_launches_nodes: Nodes that are in the launch queue.
            adjusted_min_workers: Nodes to launch to satisfy
                min_workers and request_resources(). This overrides the launch
                limits since the user is hinting to immediately scale up to
                this size.
            placement_group_nodes: Nodes to launch for placement groups.
                This overrides the launch concurrency limits.
        Returns:
            Dict[NodeType, int]: Maximum number of nodes to launch for each
                node type.
        """
    updated_nodes_to_launch = {}
    running_nodes, pending_nodes = self._separate_running_and_pending_nodes(non_terminated_nodes, connected_nodes)
    for node_type in to_launch:
        max_allowed_pending_nodes = max(UPSCALING_INITIAL_NUM_NODES, int(self.upscaling_speed * max(running_nodes[node_type], 1)))
        total_pending_nodes = pending_launches_nodes.get(node_type, 0) + pending_nodes[node_type]
        upper_bound = max(max_allowed_pending_nodes - total_pending_nodes, adjusted_min_workers.get(node_type, 0) + placement_group_nodes.get(node_type, 0))
        if upper_bound > 0:
            updated_nodes_to_launch[node_type] = min(upper_bound, to_launch[node_type])
    return updated_nodes_to_launch