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
def _parse_cluster_resource_usage(cls, state: ClusterResourceState) -> List[ResourceUsage]:
    """
        Parse the cluster resource usage from the cluster resource state.
        Args:
            state: the cluster resource state
        Returns:
            cluster_resource_usage: the cluster resource usage
        """
    cluster_resource_usage = defaultdict(ResourceUsage)
    for node_state in state.node_states:
        if node_state.status != NodeStatus.DEAD:
            cluster_resource_usage = cls._parse_node_resource_usage(node_state, cluster_resource_usage)
    return list(cluster_resource_usage.values())