import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
@dataclass
class ResourceDemandSummary:
    placement_group_demand: List[PlacementGroupResourceDemand] = field(default_factory=list)
    ray_task_actor_demand: List[RayTaskActorDemand] = field(default_factory=list)
    cluster_constraint_demand: List[ClusterConstraintDemand] = field(default_factory=list)