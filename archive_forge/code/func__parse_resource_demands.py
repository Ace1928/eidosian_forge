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
def _parse_resource_demands(cls, state: ClusterResourceState) -> List[ResourceDemand]:
    """
        Parse the resource demands from the cluster resource state.
        Args:
            state: the cluster resource state
        Returns:
            resource_demands: the resource demands
        """
    task_actor_demand = []
    pg_demand = []
    constraint_demand = []
    for request_count in state.pending_resource_requests:
        demand = RayTaskActorDemand(bundles_by_count=[ResourceRequestByCount(request_count.request.resources_bundle, request_count.count)])
        task_actor_demand.append(demand)
    for gang_request in state.pending_gang_resource_requests:
        demand = PlacementGroupResourceDemand(bundles_by_count=cls._aggregate_resource_requests_by_shape(gang_request.requests), details=gang_request.details)
        pg_demand.append(demand)
    for constraint_request in state.cluster_resource_constraints:
        demand = ClusterConstraintDemand(bundles_by_count=[ResourceRequestByCount(bundle=dict(r.request.resources_bundle.items()), count=r.count) for r in constraint_request.min_bundles])
        constraint_demand.append(demand)
    return ResourceDemandSummary(ray_task_actor_demand=task_actor_demand, placement_group_demand=pg_demand, cluster_constraint_demand=constraint_demand)