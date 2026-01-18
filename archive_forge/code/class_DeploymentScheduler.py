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
class DeploymentScheduler(ABC):
    """A centralized scheduler for all Serve deployments.

    It makes a batch of scheduling decisions in each update cycle.
    """

    @abstractmethod
    def on_deployment_created(self, deployment_id: DeploymentID, scheduling_policy: SpreadDeploymentSchedulingPolicy) -> None:
        """Called whenever a new deployment is created."""
        raise NotImplementedError

    @abstractmethod
    def on_deployment_deleted(self, deployment_id: DeploymentID) -> None:
        """Called whenever a deployment is deleted."""
        raise NotImplementedError

    @abstractmethod
    def on_replica_stopping(self, deployment_id: DeploymentID, replica_name: str) -> None:
        """Called whenever a deployment replica is being stopped."""
        raise NotImplementedError

    @abstractmethod
    def on_replica_running(self, deployment_id: DeploymentID, replica_name: str, node_id: str) -> None:
        """Called whenever a deployment replica is running with a known node id."""
        raise NotImplementedError

    @abstractmethod
    def on_replica_recovering(self, deployment_id: DeploymentID, replica_name: str) -> None:
        """Called whenever a deployment replica is recovering."""
        raise NotImplementedError

    @abstractmethod
    def schedule(self, upscales: Dict[DeploymentID, List[ReplicaSchedulingRequest]], downscales: Dict[DeploymentID, DeploymentDownscaleRequest]) -> Dict[DeploymentID, Set[str]]:
        """Called for each update cycle to do batch scheduling.

        Args:
            upscales: a dict of deployment name to a list of replicas to schedule.
            downscales: a dict of deployment name to a downscale request.

        Returns:
            The name of replicas to stop for each deployment.
        """
        raise NotImplementedError