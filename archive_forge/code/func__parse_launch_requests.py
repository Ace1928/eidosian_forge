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
def _parse_launch_requests(cls, state: AutoscalingState) -> Tuple[List[LaunchRequest], List[LaunchRequest]]:
    """
        Parse the launch requests from the autoscaling state.
        Args:
            state: the autoscaling state, empty if there's no autoscaling state
                being reported.
        Returns:
            pending_launches: the list of pending launches
            failed_launches: the list of failed launches
        """
    pending_launches = []
    for pending_request in state.pending_instance_requests:
        launch = LaunchRequest(instance_type_name=pending_request.instance_type_name, ray_node_type_name=pending_request.ray_node_type_name, count=pending_request.count, state=LaunchRequest.Status.PENDING, request_ts_s=pending_request.request_ts)
        pending_launches.append(launch)
    failed_launches = []
    for failed_request in state.failed_instance_requests:
        launch = LaunchRequest(instance_type_name=failed_request.instance_type_name, ray_node_type_name=failed_request.ray_node_type_name, count=failed_request.count, state=LaunchRequest.Status.FAILED, request_ts_s=failed_request.start_ts, details=failed_request.reason, failed_ts_s=failed_request.failed_ts)
        failed_launches.append(launch)
    return (pending_launches, failed_launches)