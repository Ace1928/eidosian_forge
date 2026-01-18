from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from itertools import chain
from typing import Any, Dict, List, Tuple
import ray
from ray._private.ray_constants import AUTOSCALER_NAMESPACE, AUTOSCALER_V2_ENABLED_KEY
from ray._private.utils import binary_to_hex
from ray.autoscaler._private.autoscaler import AutoscalerSummary
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.util import LoadMetricsSummary, format_info_string
from ray.autoscaler.v2.schema import (
from ray.core.generated.autoscaler_pb2 import (
from ray.core.generated.autoscaler_pb2 import (
from ray.core.generated.instance_manager_pb2 import Instance
from ray.experimental.internal_kv import _internal_kv_get, _internal_kv_initialized
@classmethod
def _parse_nodes(cls, state: ClusterResourceState) -> Tuple[List[NodeInfo], List[NodeInfo]]:
    """
        Parse the node info from the cluster resource state.
        Args:
            state: the cluster resource state
        Returns:
            active_nodes: the list of non-idle nodes
            idle_nodes: the list of idle nodes
            dead_nodes: the list of dead nodes
        """
    active_nodes = []
    dead_nodes = []
    idle_nodes = []
    for node_state in state.node_states:
        node_id = binary_to_hex(node_state.node_id)
        if len(node_state.ray_node_type_name) == 0:
            ray_node_type_name = f'node_{node_id}'
        else:
            ray_node_type_name = node_state.ray_node_type_name
        node_resource_usage = None
        failure_detail = None
        if node_state.status == NodeStatus.DEAD:
            failure_detail = NODE_DEATH_CAUSE_RAYLET_DIED
        else:
            usage = defaultdict(ResourceUsage)
            usage = cls._parse_node_resource_usage(node_state, usage)
            node_resource_usage = NodeUsage(usage=list(usage.values()), idle_time_ms=node_state.idle_duration_ms if node_state.status == NodeStatus.IDLE else 0)
        node_info = NodeInfo(instance_type_name=node_state.instance_type_name, node_status=NodeStatus.Name(node_state.status), node_id=binary_to_hex(node_state.node_id), ip_address=node_state.node_ip_address, ray_node_type_name=ray_node_type_name, instance_id=node_state.instance_id, resource_usage=node_resource_usage, failure_detail=failure_detail, node_activity=node_state.node_activity)
        if node_state.status == NodeStatus.DEAD:
            dead_nodes.append(node_info)
        elif node_state.status == NodeStatus.IDLE:
            idle_nodes.append(node_info)
        else:
            active_nodes.append(node_info)
    return (active_nodes, idle_nodes, dead_nodes)