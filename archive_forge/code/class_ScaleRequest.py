import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.util import NodeID, NodeIP, NodeKind, NodeStatus, NodeType
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
@dataclass
class ScaleRequest:
    """Stores desired scale computed by the autoscaler.

    Attributes:
        desired_num_workers: Map of worker NodeType to desired number of workers of
            that type.
        workers_to_delete: List of ids of nodes that should be removed.
    """
    desired_num_workers: Dict[NodeType, int] = field(default_factory=dict)
    workers_to_delete: Set[NodeID] = field(default_factory=set)