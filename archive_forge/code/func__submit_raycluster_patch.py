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
def _submit_raycluster_patch(self, patch_payload: List[Dict[str, Any]]):
    """Submits a patch to modify a RayCluster CR."""
    path = 'rayclusters/{}'.format(self.cluster_name)
    self._patch(path, patch_payload)