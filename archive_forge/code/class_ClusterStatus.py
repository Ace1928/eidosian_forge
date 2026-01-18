import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
@dataclass
class ClusterStatus:
    active_nodes: List[NodeInfo] = field(default_factory=list)
    idle_nodes: List[NodeInfo] = field(default_factory=list)
    pending_launches: List[LaunchRequest] = field(default_factory=list)
    failed_launches: List[LaunchRequest] = field(default_factory=list)
    pending_nodes: List[NodeInfo] = field(default_factory=list)
    failed_nodes: List[NodeInfo] = field(default_factory=list)
    cluster_resource_usage: List[ResourceUsage] = field(default_factory=list)
    resource_demands: ResourceDemandSummary = field(default_factory=ResourceDemandSummary)
    stats: Stats = field(default_factory=Stats)

    def total_resources(self) -> Dict[str, float]:
        return {r.resource_name: r.total for r in self.cluster_resource_usage}

    def available_resources(self) -> Dict[str, float]:
        return {r.resource_name: r.total - r.used for r in self.cluster_resource_usage}