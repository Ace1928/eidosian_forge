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
def _wait_for_deployment_healthy(self, name: str, timeout_s: int=-1):
    """Waits for the named deployment to enter "HEALTHY" status.

        Raises RuntimeError if the deployment enters the "UNHEALTHY" status
        instead.

        Raises TimeoutError if this doesn't happen before timeout_s.
        """
    start = time.time()
    while time.time() - start < timeout_s or timeout_s < 0:
        status_bytes = ray.get(self._controller.get_deployment_status.remote(name))
        if status_bytes is None:
            raise RuntimeError(f"Waiting for deployment {name} to be HEALTHY, but deployment doesn't exist.")
        status = DeploymentStatusInfo.from_proto(DeploymentStatusInfoProto.FromString(status_bytes))
        if status.status == DeploymentStatus.HEALTHY:
            break
        elif status.status == DeploymentStatus.UNHEALTHY:
            raise RuntimeError(f'Deployment {name} is UNHEALTHY: {status.message}')
        else:
            assert status.status == DeploymentStatus.UPDATING
        logger.debug(f'Waiting for {name} to be healthy, current status: {status.status}.')
        time.sleep(CLIENT_POLLING_INTERVAL_S)
    else:
        raise TimeoutError(f'Deployment {name} did not become HEALTHY after {timeout_s}s.')