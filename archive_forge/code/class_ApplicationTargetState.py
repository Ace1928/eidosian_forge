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
@dataclass
class ApplicationTargetState:
    """Defines target state of application.

    Target state can become inconsistent if the code version doesn't
    match that of the config. When that happens, a new build app task
    should be kicked off to reconcile the inconsistency.

    deployment_infos: map of deployment name to deployment info. This is
      - None if a config was deployed but the app hasn't finished
        building yet,
      - An empty dict if the app is deleting.
    code_version: Code version of all deployments in target state. None
        if application was deployed through serve.run.
    config: application config deployed by user. None if application was
        deployed through serve.run.
    target_capacity: the target_capacity to use when adjusting num_replicas.
    target_capacity_direction: the scale direction to use when
        running the Serve autoscaler.
    deleting: whether the application is being deleted.
    """
    deployment_infos: Optional[Dict[str, DeploymentInfo]]
    code_version: Optional[str]
    config: Optional[ServeApplicationSchema]
    target_capacity: Optional[float]
    target_capacity_direction: Optional[TargetCapacityDirection]
    deleting: bool