import copy
import logging
import math
import operator
import os
import queue
import subprocess
import threading
import time
from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union
import yaml
import ray
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.event_summarizer import EventSummarizer
from ray.autoscaler._private.legacy_info_string import legacy_log_info_string
from ray.autoscaler._private.load_metrics import LoadMetrics
from ray.autoscaler._private.local.node_provider import (
from ray.autoscaler._private.node_launcher import BaseNodeLauncher, NodeLauncher
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.node_tracker import NodeTracker
from ray.autoscaler._private.prom_metrics import AutoscalerPrometheusMetrics
from ray.autoscaler._private.providers import _get_node_provider
from ray.autoscaler._private.resource_demand_scheduler import (
from ray.autoscaler._private.updater import NodeUpdaterThread
from ray.autoscaler._private.util import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
from ray.exceptions import RpcError
def _get_nodes_needed_for_request_resources(self, sorted_node_ids: List[NodeID]) -> FrozenSet[NodeID]:
    """Returns the nodes NOT allowed to terminate due to request_resources().

        Args:
            sorted_node_ids: the node ids sorted based on last used (LRU last).

        Returns:
            FrozenSet[NodeID]: a set of nodes (node ids) that
            we should NOT terminate.
        """
    assert self.provider
    nodes_not_allowed_to_terminate: Set[NodeID] = set()
    static_node_resources: Dict[NodeIP, ResourceDict] = self.load_metrics.get_static_node_resources_by_ip()
    head_node_resources: ResourceDict = copy.deepcopy(self.available_node_types[self.config['head_node_type']]['resources'])
    if not head_node_resources:
        head_node_ip = self.provider.internal_ip(self.non_terminated_nodes.head_id)
        head_node_resources = static_node_resources.get(head_node_ip, {})
    max_node_resources: List[ResourceDict] = [head_node_resources]
    resource_demand_vector_worker_node_ids = []
    for node_id in sorted_node_ids:
        tags = self.provider.node_tags(node_id)
        if TAG_RAY_USER_NODE_TYPE in tags:
            node_type = tags[TAG_RAY_USER_NODE_TYPE]
            node_resources: ResourceDict = copy.deepcopy(self.available_node_types[node_type]['resources'])
            if not node_resources:
                node_ip = self.provider.internal_ip(node_id)
                node_resources = static_node_resources.get(node_ip, {})
            max_node_resources.append(node_resources)
            resource_demand_vector_worker_node_ids.append(node_id)
    used_resource_requests: List[ResourceDict]
    _, used_resource_requests = get_bin_pack_residual(max_node_resources, self.load_metrics.get_resource_requests())
    max_node_resources.pop(0)
    used_resource_requests.pop(0)
    for i, node_id in enumerate(resource_demand_vector_worker_node_ids):
        if used_resource_requests[i] == max_node_resources[i] and max_node_resources[i]:
            pass
        else:
            nodes_not_allowed_to_terminate.add(node_id)
    return frozenset(nodes_not_allowed_to_terminate)