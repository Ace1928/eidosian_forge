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
def _determine_app_status(self) -> Tuple[ApplicationStatus, str]:
    """Check deployment statuses and target state, and determine the
        corresponding application status.

        Returns:
            Status (ApplicationStatus):
                RUNNING: all deployments are healthy or autoscaling.
                DEPLOYING: there is one or more updating deployments,
                    and there are no unhealthy deployments.
                DEPLOY_FAILED: one or more deployments became unhealthy
                    while the application was deploying.
                UNHEALTHY: one or more deployments became unhealthy
                    while the application was running.
                DELETING: the application is being deleted.
            Error message (str):
                Non-empty string if status is DEPLOY_FAILED or UNHEALTHY
        """
    if self._target_state.deleting:
        return (ApplicationStatus.DELETING, '')
    num_healthy_deployments = 0
    num_autoscaling_deployments = 0
    num_updating_deployments = 0
    num_manually_scaling_deployments = 0
    unhealthy_deployment_names = []
    for deployment_status in self.get_deployments_statuses():
        if deployment_status.status == DeploymentStatus.UNHEALTHY:
            unhealthy_deployment_names.append(deployment_status.name)
        elif deployment_status.status == DeploymentStatus.HEALTHY:
            num_healthy_deployments += 1
        elif deployment_status.status_trigger == DeploymentStatusTrigger.AUTOSCALING:
            num_autoscaling_deployments += 1
        elif deployment_status.status == DeploymentStatus.UPDATING:
            num_updating_deployments += 1
        elif deployment_status.status in [DeploymentStatus.UPSCALING, DeploymentStatus.DOWNSCALING] and deployment_status.status_trigger == DeploymentStatusTrigger.CONFIG_UPDATE_STARTED:
            num_manually_scaling_deployments += 1
        else:
            raise RuntimeError(f'Found deployment with unexpected status {deployment_status.status} and status trigger {deployment_status.status_trigger}.')
    if len(unhealthy_deployment_names):
        status_msg = f'The deployments {unhealthy_deployment_names} are UNHEALTHY.'
        if self._status in [ApplicationStatus.DEPLOYING, ApplicationStatus.DEPLOY_FAILED]:
            return (ApplicationStatus.DEPLOY_FAILED, status_msg)
        else:
            return (ApplicationStatus.UNHEALTHY, status_msg)
    elif num_updating_deployments + num_manually_scaling_deployments > 0:
        return (ApplicationStatus.DEPLOYING, '')
    else:
        assert num_healthy_deployments + num_autoscaling_deployments == len(self.target_deployments)
        return (ApplicationStatus.RUNNING, '')