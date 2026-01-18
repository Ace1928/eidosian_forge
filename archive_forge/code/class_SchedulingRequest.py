import copy
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from ray._private.protobuf_compat import message_to_dict
from ray.autoscaler._private.resource_demand_scheduler import UtilizationScore
from ray.autoscaler.v2.schema import NodeType
from ray.autoscaler.v2.utils import is_pending, resource_requests_by_count
from ray.core.generated.autoscaler_pb2 import (
from ray.core.generated.instance_manager_pb2 import Instance
@dataclass
class SchedulingRequest:
    cluster_config: ClusterConfig
    resource_requests: List[ResourceRequestByCount] = field(default_factory=list)
    gang_resource_requests: List[GangResourceRequest] = field(default_factory=list)
    cluster_resource_constraints: List[ClusterResourceConstraint] = field(default_factory=list)
    current_nodes: List[NodeState] = field(default_factory=list)
    current_instances: List[Instance] = field(default_factory=list)