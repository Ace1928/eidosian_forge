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
def _recover_state_from_checkpoint(self):
    deployment_time, serve_config, target_capacity_direction = self._read_config_checkpoint()
    self._target_capacity_direction = target_capacity_direction
    if serve_config is not None:
        logger.info('Recovered config from checkpoint.', extra={'log_to_stderr': False})
        self.deploy_config(serve_config, deployment_time=deployment_time)