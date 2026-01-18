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
def _get_head_and_workers(self, nodes: List[NodeID]) -> Tuple[NodeID, List[NodeID]]:
    """Returns the head node's id and the list of all worker node ids,
        given a list `nodes` of all node ids in the cluster.
        """
    head_id, worker_ids = (None, [])
    for node in nodes:
        tags = self.provider.node_tags(node)
        if tags[TAG_RAY_NODE_KIND] == NODE_KIND_HEAD:
            head_id = node
        elif tags[TAG_RAY_NODE_KIND] == NODE_KIND_WORKER:
            worker_ids.append(node)
    return (head_id, worker_ids)