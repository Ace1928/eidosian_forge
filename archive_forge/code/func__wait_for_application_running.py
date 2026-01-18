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
def _wait_for_application_running(self, name: str, timeout_s: int=-1):
    """Waits for the named application to enter "RUNNING" status.

        Raises:
            RuntimeError: if the application enters the "DEPLOY_FAILED" status instead.
            TimeoutError: if this doesn't happen before timeout_s.
        """
    start = time.time()
    while time.time() - start < timeout_s or timeout_s < 0:
        status_bytes = ray.get(self._controller.get_serve_status.remote(name))
        if status_bytes is None:
            raise RuntimeError(f"Waiting for application {name} to be RUNNING, but application doesn't exist.")
        status = StatusOverview.from_proto(StatusOverviewProto.FromString(status_bytes))
        if status.app_status.status == ApplicationStatus.RUNNING:
            break
        elif status.app_status.status == ApplicationStatus.DEPLOY_FAILED:
            raise RuntimeError(f'Deploying application {name} failed: {status.app_status.message}')
        logger.debug(f'Waiting for {name} to be RUNNING, current status: {status.app_status.status}.')
        time.sleep(CLIENT_POLLING_INTERVAL_S)
    else:
        raise TimeoutError(f'Application {name} did not become RUNNING after {timeout_s}s.')