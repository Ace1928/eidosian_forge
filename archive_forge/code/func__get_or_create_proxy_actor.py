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
@staticmethod
def _get_or_create_proxy_actor(config: HTTPOptions, grpc_options: gRPCOptions, controller_name: str, name: str, node_id: str, node_ip_address: str, port: int, logging_config: LoggingConfig, proxy_actor_class: Type[ProxyActor]=ProxyActor) -> ProxyWrapper:
    """Helper to start or reuse existing proxy.

        Takes the name of the proxy, the node id, and the node ip address, and look up
        or creates a new ProxyActor actor handle for the proxy.
        """
    proxy = None
    try:
        proxy = ray.get_actor(name, namespace=SERVE_NAMESPACE)
    except ValueError:
        logger.info(f"Starting proxy with name '{name}' on node '{node_id}' listening on '{config.host}:{port}'", extra={'log_to_stderr': False})
    proxy = proxy or proxy_actor_class.options(num_cpus=config.num_cpus, name=name, namespace=SERVE_NAMESPACE, lifetime='detached', max_concurrency=ASYNC_CONCURRENCY, max_restarts=0, scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False)).remote(config.host, port, config.root_path, controller_name=controller_name, node_ip_address=node_ip_address, node_id=node_id, http_middlewares=config.middlewares, request_timeout_s=config.request_timeout_s, keep_alive_timeout_s=config.keep_alive_timeout_s, grpc_options=grpc_options, logging_config=logging_config)
    return proxy