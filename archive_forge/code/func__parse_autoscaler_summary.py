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
def _parse_autoscaler_summary(cls, data: ClusterStatus) -> AutoscalerSummary:
    active_nodes = _count_by(data.active_nodes, 'ray_node_type_name')
    idle_nodes = _count_by(data.idle_nodes, 'ray_node_type_name')
    pending_launches = _count_by(data.pending_launches, 'ray_node_type_name')
    pending_nodes = []
    for node in data.pending_nodes:
        pending_nodes.append((node.ip_address, node.ray_node_type_name, node.details))
    failed_nodes = []
    for node in data.failed_nodes:
        failed_nodes.append((node.ip_address, node.ray_node_type_name))
    node_type_mapping = {}
    for node in chain(data.active_nodes, data.idle_nodes):
        node_type_mapping[node.ip_address] = node.ray_node_type_name
    node_availabilities = {}
    for failed_launch in data.failed_launches:
        node_availabilities[failed_launch.ray_node_type_name] = NodeAvailabilityRecord(node_type=failed_launch.ray_node_type_name, is_available=False, last_checked_timestamp=failed_launch.request_ts_s, unavailable_node_information=UnavailableNodeInformation(category='LaunchFailed', description=failed_launch.details))
    node_availabilities = NodeAvailabilitySummary(node_availabilities=node_availabilities)
    node_activities = {node.node_id: (node.ip_address, node.node_activity) for node in data.active_nodes}
    return AutoscalerSummary(active_nodes=active_nodes, idle_nodes=idle_nodes, pending_launches=pending_launches, pending_nodes=pending_nodes, failed_nodes=failed_nodes, pending_resources={}, node_type_mapping=node_type_mapping, node_availability_summary=node_availabilities, node_activities=node_activities)