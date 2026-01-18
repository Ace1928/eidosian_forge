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
def _aggregate_resource_requests_by_shape(cls, requests: List[ResourceRequest]) -> List[ResourceRequestByCount]:
    """
        Aggregate resource requests by shape.
        Args:
            requests: the list of resource requests
        Returns:
            resource_requests_by_count: the aggregated resource requests by count
        """
    resource_requests_by_count = defaultdict(int)
    for request in requests:
        bundle = frozenset(request.resources_bundle.items())
        resource_requests_by_count[bundle] += 1
    return [ResourceRequestByCount(dict(bundle), count) for bundle, count in resource_requests_by_count.items()]