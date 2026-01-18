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
class ReplicaSchedulingRequest:
    """Request to schedule a single replica.

    The scheduler is responsible for scheduling
    based on the deployment scheduling policy.
    """
    deployment_id: DeploymentID
    replica_name: str
    actor_def: ray.actor.ActorClass
    actor_resources: Dict
    actor_options: Dict
    actor_init_args: Tuple
    on_scheduled: Callable
    placement_group_bundles: Optional[List[Dict[str, float]]] = None
    placement_group_strategy: Optional[str] = None
    max_replicas_per_node: Optional[int] = None