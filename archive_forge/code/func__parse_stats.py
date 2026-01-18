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
def _parse_stats(cls, reply: GetClusterStatusReply, stats: Stats) -> Stats:
    """
        Parse the stats from the get cluster status reply.
        Args:
            reply: the get cluster status reply
            stats: the stats
        Returns:
            stats: the parsed stats
        """
    stats = deepcopy(stats)
    stats.gcs_request_time_s = stats.gcs_request_time_s
    stats.autoscaler_version = str(reply.autoscaling_state.autoscaler_state_version)
    stats.cluster_resource_state_version = str(reply.cluster_resource_state.cluster_resource_state_version)
    return stats