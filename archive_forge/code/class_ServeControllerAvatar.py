import asyncio
import logging
import marshal
import os
import pickle
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import ray
from ray._private.resource_spec import HEAD_NODE_RESOURCE_NAME
from ray._private.utils import run_background_task
from ray._raylet import GcsClient
from ray.actor import ActorHandle
from ray.serve._private.application_state import ApplicationStateManager
from ray.serve._private.common import (
from ray.serve._private.constants import (
from ray.serve._private.default_impl import create_cluster_node_info_cache
from ray.serve._private.deploy_utils import deploy_args_to_deployment_info
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.deployment_state import DeploymentStateManager
from ray.serve._private.endpoint_state import EndpointState
from ray.serve._private.logging_utils import (
from ray.serve._private.long_poll import LongPollHost, LongPollNamespace
from ray.serve._private.proxy_state import ProxyStateManager
from ray.serve._private.storage.kv_store import RayInternalKVStore
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import (
from ray.serve.config import HTTPOptions, gRPCOptions
from ray.serve.generated.serve_pb2 import (
from ray.serve.generated.serve_pb2 import EndpointInfo as EndpointInfoProto
from ray.serve.generated.serve_pb2 import EndpointSet
from ray.serve.schema import (
from ray.util import metrics
@ray.remote(num_cpus=0)
class ServeControllerAvatar:
    """A hack that proxy the creation of async actors from Java.

    To be removed after https://github.com/ray-project/ray/pull/26037

    Java api can not support python async actor. If we use java api create
    python async actor. The async init method won't be executed. The async
    method will fail with pickle error. And the run_control_loop of controller
    actor can't be executed too. We use this proxy actor create python async
    actor to avoid the above problem.
    """

    def __init__(self, controller_name: str, http_proxy_port: int=8000):
        try:
            self._controller = ray.get_actor(controller_name, namespace=SERVE_NAMESPACE)
        except ValueError:
            self._controller = None
        if self._controller is None:
            http_config = HTTPOptions()
            logging_config = LoggingConfig()
            http_config.port = http_proxy_port
            self._controller = ServeController.options(num_cpus=0, name=controller_name, lifetime='detached', max_restarts=-1, max_task_retries=-1, resources={HEAD_NODE_RESOURCE_NAME: 0.001}, namespace=SERVE_NAMESPACE, max_concurrency=CONTROLLER_MAX_CONCURRENCY).remote(controller_name, http_config=http_config, global_logging_config=logging_config)

    def check_alive(self) -> None:
        """No-op to check if this actor is alive."""
        return