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
def _schedule_spread_deployment(self, deployment_id: DeploymentID) -> None:
    for pending_replica_name in list(self._pending_replicas[deployment_id].keys()):
        replica_scheduling_request = self._pending_replicas[deployment_id][pending_replica_name]
        placement_group = None
        if replica_scheduling_request.placement_group_bundles is not None:
            strategy = replica_scheduling_request.placement_group_strategy if replica_scheduling_request.placement_group_strategy else 'PACK'
            placement_group = ray.util.placement_group(replica_scheduling_request.placement_group_bundles, strategy=strategy, lifetime='detached', name=replica_scheduling_request.actor_options['name'])
            scheduling_strategy = PlacementGroupSchedulingStrategy(placement_group=placement_group, placement_group_capture_child_tasks=True)
        else:
            scheduling_strategy = 'SPREAD'
        actor_options = copy.copy(replica_scheduling_request.actor_options)
        if replica_scheduling_request.max_replicas_per_node is not None:
            if 'resources' not in actor_options:
                actor_options['resources'] = {}
            actor_options['resources'][f'{ray._raylet.IMPLICIT_RESOURCE_PREFIX}{deployment_id.app}:{deployment_id.name}'] = 1.0 / replica_scheduling_request.max_replicas_per_node
        actor_handle = replica_scheduling_request.actor_def.options(scheduling_strategy=scheduling_strategy, **actor_options).remote(*replica_scheduling_request.actor_init_args)
        del self._pending_replicas[deployment_id][pending_replica_name]
        self._launching_replicas[deployment_id][pending_replica_name] = None
        replica_scheduling_request.on_scheduled(actor_handle, placement_group=placement_group)