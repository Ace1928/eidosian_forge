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
def _add_min_workers_nodes(node_resources: List[ResourceDict], node_type_counts: Dict[NodeType, int], node_types: Dict[NodeType, NodeTypeConfigDict], max_workers: int, head_node_type: NodeType, ensure_min_cluster_size: List[ResourceDict], utilization_scorer: Callable[[NodeResources, ResourceDemands, str], Optional[UtilizationScore]]) -> (List[ResourceDict], Dict[NodeType, int], Dict[NodeType, int]):
    """Updates resource demands to respect the min_workers and
    request_resources() constraints.

    Args:
        node_resources: Resources of exisiting nodes already launched/pending.
        node_type_counts: Counts of existing nodes already launched/pending.
        node_types: Node types config.
        max_workers: global max_workers constaint.
        ensure_min_cluster_size: resource demands from request_resources().
        utilization_scorer: A function that, given a node
            type, its resources, and resource demands, returns what its
            utilization would be.

    Returns:
        node_resources: The updated node resources after adding min_workers
            and request_resources() constraints per node type.
        node_type_counts: The updated node counts after adding min_workers
            and request_resources() constraints per node type.
        total_nodes_to_add_dict: The nodes to add to respect min_workers and
            request_resources() constraints.
    """
    total_nodes_to_add_dict = {}
    for node_type, config in node_types.items():
        existing = node_type_counts.get(node_type, 0)
        target = min(config.get('min_workers', 0), config.get('max_workers', 0))
        if node_type == head_node_type:
            target = target + 1
        if existing < target:
            total_nodes_to_add_dict[node_type] = target - existing
            node_type_counts[node_type] = target
            node_resources.extend([copy.deepcopy(node_types[node_type]['resources']) for _ in range(total_nodes_to_add_dict[node_type])])
    if ensure_min_cluster_size:
        max_to_add = max_workers + 1 - sum(node_type_counts.values())
        max_node_resources = []
        for node_type in node_type_counts:
            max_node_resources.extend([copy.deepcopy(node_types[node_type]['resources']) for _ in range(node_type_counts[node_type])])
        resource_requests_unfulfilled, _ = get_bin_pack_residual(max_node_resources, ensure_min_cluster_size)
        nodes_to_add_request_resources, _ = get_nodes_for(node_types, node_type_counts, head_node_type, max_to_add, resource_requests_unfulfilled, utilization_scorer=utilization_scorer)
        for node_type in nodes_to_add_request_resources:
            nodes_to_add = nodes_to_add_request_resources.get(node_type, 0)
            if nodes_to_add > 0:
                node_type_counts[node_type] = nodes_to_add + node_type_counts.get(node_type, 0)
                node_resources.extend([copy.deepcopy(node_types[node_type]['resources']) for _ in range(nodes_to_add)])
                total_nodes_to_add_dict[node_type] = nodes_to_add + total_nodes_to_add_dict.get(node_type, 0)
    return (node_resources, node_type_counts, total_nodes_to_add_dict)