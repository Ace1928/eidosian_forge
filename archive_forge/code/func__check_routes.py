import logging
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
import ray
from ray import cloudpickle
from ray._private.utils import import_attr
from ray.exceptions import RuntimeEnvSetupError
from ray.serve._private.common import (
from ray.serve._private.config import DeploymentConfig
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.deploy_utils import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.deployment_state import DeploymentStateManager
from ray.serve._private.endpoint_state import EndpointState
from ray.serve._private.storage.kv_store import KVStoreBase
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import (
from ray.serve.exceptions import RayServeException
from ray.serve.generated.serve_pb2 import DeploymentLanguage
from ray.serve.schema import DeploymentDetails, ServeApplicationSchema
from ray.types import ObjectRef
def _check_routes(self, deployment_infos: Dict[str, DeploymentInfo]) -> Tuple[str, str]:
    """Check route prefixes and docs paths of deployments in app.

        There should only be one non-null route prefix. If there is one,
        set it as the application route prefix. This function must be
        run every control loop iteration because the target config could
        be updated without kicking off a new task.

        Returns: tuple of route prefix, docs path.
        Raises: RayServeException if more than one route prefix or docs
            path is found among deployments.
        """
    num_route_prefixes = 0
    num_docs_paths = 0
    route_prefix = None
    docs_path = None
    for info in deployment_infos.values():
        if info.route_prefix is not None:
            route_prefix = info.route_prefix
            num_route_prefixes += 1
        if info.docs_path is not None:
            docs_path = info.docs_path
            num_docs_paths += 1
    if num_route_prefixes > 1:
        raise RayServeException(f'Found multiple route prefixes from application "{self._name}", Please specify only one route prefix for the application to avoid this issue.')
    if num_docs_paths > 1:
        raise RayServeException(f'Found multiple deployments in application "{self._name}" that have a docs path. This may be due to using multiple FastAPI deployments in your application. Please only include one deployment with a docs path in your application to avoid this issue.')
    return (route_prefix, docs_path)