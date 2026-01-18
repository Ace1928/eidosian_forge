import copy
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from ray._private.protobuf_compat import message_to_dict
from ray.autoscaler._private.resource_demand_scheduler import UtilizationScore
from ray.autoscaler.v2.schema import NodeType
from ray.autoscaler.v2.utils import is_pending, resource_requests_by_count
from ray.core.generated.autoscaler_pb2 import (
from ray.core.generated.instance_manager_pb2 import Instance
@staticmethod
def _compute_available_node_types(nodes: List[SchedulingNode], cluster_config: ClusterConfig) -> Dict[NodeType, int]:
    """
            Compute the number of nodes by node types available for launching based on
            the max number of workers in the config.
            Args:
                nodes: The current existing nodes.
                cluster_config: The cluster instances config.
            Returns:
                A dict of node types and the number of nodes available for launching.
            """
    node_type_available: Dict[NodeType, int] = defaultdict(int)
    node_type_existing: Dict[NodeType, int] = defaultdict(int)
    for node in nodes:
        node_type_existing[node.node_type] += 1
    for node_type, node_type_config in cluster_config.node_type_configs.items():
        node_type_available[node_type] = node_type_config.max_workers - node_type_existing.get(node_type, 0)
    return node_type_available