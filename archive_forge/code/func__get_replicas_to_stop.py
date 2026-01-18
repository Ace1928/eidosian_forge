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
def _get_replicas_to_stop(self, deployment_id: DeploymentID, max_num_to_stop: int) -> Set[str]:
    """Prioritize replicas running on a node with fewest replicas of
            all deployments.

        This algorithm helps to scale down more intelligently because it can
        relinquish nodes faster. Note that this algorithm doesn't consider
        other non-serve actors on the same node. See more at
        https://github.com/ray-project/ray/issues/20599.
        """
    replicas_to_stop = set()
    pending_launching_recovering_replicas = set().union(self._pending_replicas[deployment_id].keys(), self._launching_replicas[deployment_id].keys(), self._recovering_replicas[deployment_id])
    for pending_launching_recovering_replica in pending_launching_recovering_replicas:
        if len(replicas_to_stop) == max_num_to_stop:
            return replicas_to_stop
        else:
            replicas_to_stop.add(pending_launching_recovering_replica)
    node_to_running_replicas_of_target_deployment = defaultdict(set)
    for running_replica, node_id in self._running_replicas[deployment_id].items():
        node_to_running_replicas_of_target_deployment[node_id].add(running_replica)
    node_to_num_running_replicas_of_all_deployments = {}
    for _, running_replicas in self._running_replicas.items():
        for running_replica, node_id in running_replicas.items():
            node_to_num_running_replicas_of_all_deployments[node_id] = node_to_num_running_replicas_of_all_deployments.get(node_id, 0) + 1

    def key(node_and_num_running_replicas_of_all_deployments):
        return node_and_num_running_replicas_of_all_deployments[1] if node_and_num_running_replicas_of_all_deployments[0] != self._head_node_id else sys.maxsize
    for node_id, _ in sorted(node_to_num_running_replicas_of_all_deployments.items(), key=key):
        if node_id not in node_to_running_replicas_of_target_deployment:
            continue
        for running_replica in node_to_running_replicas_of_target_deployment[node_id]:
            if len(replicas_to_stop) == max_num_to_stop:
                return replicas_to_stop
            else:
                replicas_to_stop.add(running_replica)
    return replicas_to_stop