import copy
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple
import ray
from ray.serve._private.cluster_node_info_cache import ClusterNodeInfoCache
from ray.serve._private.common import DeploymentID
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
@dataclass
class DeploymentDownscaleRequest:
    """Request to stop a certain number of replicas.

    The scheduler is responsible for
    choosing the replicas to stop.
    """
    deployment_id: DeploymentID
    num_to_stop: int