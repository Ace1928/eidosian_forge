from abc import ABC, abstractmethod
from typing import List, Optional, Set, Tuple
import ray
from ray._raylet import GcsClient
from ray.serve._private.constants import RAY_GCS_RPC_TIMEOUT_S
class DefaultClusterNodeInfoCache(ClusterNodeInfoCache):

    def __init__(self, gcs_client: GcsClient):
        super().__init__(gcs_client)

    def get_draining_node_ids(self) -> Set[str]:
        return set()

    def get_node_az(self, node_id: str) -> Optional[str]:
        """Get availability zone of a node."""
        return None