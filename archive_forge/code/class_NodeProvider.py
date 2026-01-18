import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Set
from ray.autoscaler._private.node_launcher import BaseNodeLauncher
from ray.autoscaler.node_provider import NodeProvider as NodeProviderV1
from ray.autoscaler.tags import TAG_RAY_USER_NODE_TYPE
from ray.autoscaler.v2.instance_manager.config import NodeProviderConfig
from ray.core.generated.instance_manager_pb2 import Instance
class NodeProvider(metaclass=ABCMeta):
    """NodeProvider defines the interface for
    interacting with cloud provider, such as AWS, GCP, Azure, etc.
    """

    @abstractmethod
    def create_nodes(self, instance_type_name: str, count: int) -> List[str]:
        """Create new nodes synchronously, returns all non-terminated nodes in the cluster.
        Note that create_nodes could fail partially.
        """
        pass

    @abstractmethod
    def terminate_node(self, cloud_instance_id: str) -> None:
        """
        Terminate nodes asynchronously, returns immediately."""
        pass

    @abstractmethod
    def get_non_terminated_nodes(self) -> Dict[str, Instance]:
        """Get all non-terminated nodes in the cluster"""
        pass

    @abstractmethod
    def get_nodes_by_cloud_instance_id(self, cloud_instance_ids: List[str]) -> Dict[str, Instance]:
        """Get nodes by node ids, including terminated nodes"""
        pass

    @abstractmethod
    def is_readonly(self) -> bool:
        return False