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
Prioritize replicas running on a node with fewest replicas of
            all deployments.

        This algorithm helps to scale down more intelligently because it can
        relinquish nodes faster. Note that this algorithm doesn't consider
        other non-serve actors on the same node. See more at
        https://github.com/ray-project/ray/issues/20599.
        