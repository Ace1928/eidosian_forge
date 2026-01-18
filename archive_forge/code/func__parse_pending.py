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
def _parse_pending(cls, state: AutoscalingState) -> List[NodeInfo]:
    """
        Parse the pending requests/nodes from the autoscaling state.
        Args:
            state: the autoscaling state, empty if there's no autoscaling state
                being reported.
        Returns:
            pending_nodes: the list of pending nodes
        """
    pending_nodes = []
    for pending_node in state.pending_instances:
        pending_nodes.append(NodeInfo(instance_type_name=pending_node.instance_type_name, ray_node_type_name=pending_node.ray_node_type_name, details=pending_node.details, instance_id=pending_node.instance_id, ip_address=pending_node.ip_address))
    return pending_nodes