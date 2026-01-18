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
def _resource_based_utilization_scorer(node_resources: ResourceDict, resources: List[ResourceDict], *, node_availability_summary: NodeAvailabilitySummary) -> Optional[Tuple[bool, int, float, float]]:
    remaining = copy.deepcopy(node_resources)
    fittable = []
    resource_types = set()
    for r in resources:
        for k, v in r.items():
            if v > 0:
                resource_types.add(k)
        if _fits(remaining, r):
            fittable.append(r)
            _inplace_subtract(remaining, r)
    if not fittable:
        return None
    util_by_resources = []
    num_matching_resource_types = 0
    for k, v in node_resources.items():
        if v < 1:
            continue
        if k in resource_types:
            num_matching_resource_types += 1
        util = (v - remaining[k]) / v
        util_by_resources.append(v * util ** 3)
    if not util_by_resources:
        return None
    gpu_ok = True
    if AUTOSCALER_CONSERVE_GPU_NODES:
        is_gpu_node = 'GPU' in node_resources and node_resources['GPU'] > 0
        any_gpu_task = any(('GPU' in r for r in resources))
        if is_gpu_node and (not any_gpu_task):
            gpu_ok = False
    return (gpu_ok, num_matching_resource_types, min(util_by_resources), float(sum(util_by_resources)) / len(util_by_resources))