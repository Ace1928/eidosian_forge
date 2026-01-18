import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.util import NodeID, NodeIP, NodeKind, NodeStatus, NodeType
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def _cur_num_workers(self, node_data_dict: Dict[str, Any]):
    num_workers_dict = defaultdict(int)
    for node_data in node_data_dict.values():
        if node_data.kind == NODE_KIND_HEAD:
            continue
        num_workers_dict[node_data.type] += 1
    return num_workers_dict