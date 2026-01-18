import json
import logging
import os
import random
import traceback
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Type
import ray
from ray.actor import ActorHandle
from ray.exceptions import RayActorError
from ray.serve._private.cluster_node_info_cache import ClusterNodeInfoCache
from ray.serve._private.common import NodeId, ProxyStatus
from ray.serve._private.constants import (
from ray.serve._private.proxy import ProxyActor
from ray.serve._private.utils import Timer, TimerBase, format_actor_name
from ray.serve.config import DeploymentMode, HTTPOptions, gRPCOptions
from ray.serve.schema import LoggingConfig, ProxyDetails
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def _get_target_nodes(self, proxy_nodes) -> List[Tuple[str, str]]:
    """Return the list of (node_id, ip_address) to deploy HTTP and gRPC servers
        on."""
    location = self._config.location
    if location == DeploymentMode.NoServer:
        return []
    target_nodes = [(node_id, ip_address) for node_id, ip_address in self._cluster_node_info_cache.get_alive_nodes() if node_id in proxy_nodes]
    if location == DeploymentMode.HeadOnly:
        nodes = [(node_id, ip_address) for node_id, ip_address in target_nodes if node_id == self._head_node_id]
        assert len(nodes) == 1, f'Head node not found! Head node id: {self._head_node_id}, all nodes: {target_nodes}.'
        return nodes
    return target_nodes