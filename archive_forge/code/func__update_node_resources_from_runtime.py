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
def _update_node_resources_from_runtime(self, nodes: List[NodeID], max_resources_by_ip: Dict[NodeIP, ResourceDict]):
    """Update static node type resources with runtime resources

        This will update the cached static node type resources with the runtime
        resources. Because we can not know the exact autofilled memory or
        object_store_memory from config file.
        """
    need_update = len(self.node_types) != len(self.node_resource_updated)
    if not need_update:
        return
    for node_id in nodes:
        tags = self.provider.node_tags(node_id)
        if TAG_RAY_USER_NODE_TYPE not in tags:
            continue
        node_type = tags[TAG_RAY_USER_NODE_TYPE]
        if node_type in self.node_resource_updated or node_type not in self.node_types:
            continue
        ip = self.provider.internal_ip(node_id)
        runtime_resources = max_resources_by_ip.get(ip)
        if runtime_resources:
            runtime_resources = copy.deepcopy(runtime_resources)
            resources = self.node_types[node_type].get('resources', {})
            for key in ['CPU', 'GPU', 'memory', 'object_store_memory']:
                if key in runtime_resources:
                    resources[key] = runtime_resources[key]
            self.node_types[node_type]['resources'] = resources
            node_kind = tags[TAG_RAY_NODE_KIND]
            if node_kind == NODE_KIND_WORKER:
                self.node_resource_updated.add(node_type)