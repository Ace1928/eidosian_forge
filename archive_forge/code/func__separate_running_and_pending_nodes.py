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
def _separate_running_and_pending_nodes(self, non_terminated_nodes: List[NodeID], connected_nodes: List[NodeIP]) -> (Dict[NodeType, int], Dict[NodeType, int]):
    """Splits connected and non terminated nodes to pending & running."""
    running_nodes = collections.defaultdict(int)
    pending_nodes = collections.defaultdict(int)
    for node_id in non_terminated_nodes:
        tags = self.provider.node_tags(node_id)
        if TAG_RAY_USER_NODE_TYPE in tags:
            node_type = tags[TAG_RAY_USER_NODE_TYPE]
            node_ip = self.provider.internal_ip(node_id)
            if node_ip in connected_nodes:
                running_nodes[node_type] += 1
            else:
                pending_nodes[node_type] += 1
    return (running_nodes, pending_nodes)