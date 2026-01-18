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
def _start_proxy(self, name: str, node_id: str, node_ip_address: str) -> ProxyWrapper:
    """Helper to start or reuse existing proxy and wrap in the proxy actor wrapper.

        Compute the HTTP port based on `TEST_WORKER_NODE_HTTP_PORT` env var and gRPC
        port based on `TEST_WORKER_NODE_GRPC_PORT` env var. Passed all the required
        variables into the proxy actor wrapper class and return the proxy actor wrapper.
        """
    port = self._config.port
    grpc_options = self._grpc_options
    if node_id != self._head_node_id and os.getenv('TEST_WORKER_NODE_HTTP_PORT') is not None:
        logger.warning(f'`TEST_WORKER_NODE_HTTP_PORT` env var is set. Using it for worker node {node_id}.')
        port = int(os.getenv('TEST_WORKER_NODE_HTTP_PORT'))
    if node_id != self._head_node_id and os.getenv('TEST_WORKER_NODE_GRPC_PORT') is not None:
        logger.warning(f'`TEST_WORKER_NODE_GRPC_PORT` env var is set. Using it for worker node {node_id}.{int(os.getenv('TEST_WORKER_NODE_GRPC_PORT'))}')
        grpc_options.port = int(os.getenv('TEST_WORKER_NODE_GRPC_PORT'))
    return self._actor_proxy_wrapper_class(logging_config=self.logging_config, config=self._config, grpc_options=grpc_options, controller_name=self._controller_name, name=name, node_id=node_id, node_ip_address=node_ip_address, port=port, proxy_actor_class=self._proxy_actor_class)