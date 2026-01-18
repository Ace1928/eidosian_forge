import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import requests
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.util import NodeID, NodeIP, NodeKind, NodeStatus, NodeType
from ray.autoscaler.batching_node_provider import (
from ray.autoscaler.tags import (
def _worker_group_index(raycluster: Dict[str, Any], group_name: str) -> int:
    """Extract worker group index from RayCluster."""
    group_names = [spec['groupName'] for spec in raycluster['spec'].get('workerGroupSpecs', [])]
    return group_names.index(group_name)