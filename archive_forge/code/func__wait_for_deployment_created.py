import logging
import random
import time
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import ray
from ray.actor import ActorHandle
from ray.serve._private.common import (
from ray.serve._private.config import DeploymentConfig, ReplicaConfig
from ray.serve._private.constants import (
from ray.serve._private.controller import ServeController
from ray.serve._private.deploy_utils import get_deploy_args
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve.config import HTTPOptions
from ray.serve.exceptions import RayServeException
from ray.serve.generated.serve_pb2 import (
from ray.serve.generated.serve_pb2 import (
from ray.serve.generated.serve_pb2 import StatusOverview as StatusOverviewProto
from ray.serve.handle import DeploymentHandle, RayServeHandle, RayServeSyncHandle
from ray.serve.schema import LoggingConfig, ServeApplicationSchema, ServeDeploySchema
def _wait_for_deployment_created(self, deployment_name: str, app_name: str, timeout_s: int=-1):
    """Waits for the named deployment to be created.

        A deployment being created simply means that its been registered
        with the deployment state manager. The deployment state manager
        will then continue to reconcile the deployment towards its
        target state.

        Raises TimeoutError if this doesn't happen before timeout_s.
        """
    start = time.time()
    while time.time() - start < timeout_s or timeout_s < 0:
        status_bytes = ray.get(self._controller.get_deployment_status.remote(deployment_name, app_name))
        if status_bytes is not None:
            break
        logger.debug(f"Waiting for deployment '{deployment_name}' in application '{app_name}' to be created.")
        time.sleep(CLIENT_CHECK_CREATION_POLLING_INTERVAL_S)
    else:
        raise TimeoutError(f"Deployment '{deployment_name}' in application '{app_name}' did not become HEALTHY after {timeout_s}s.")