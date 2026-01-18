import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional, Union
import ray
import ray.util.serialization_addons
from ray.serve._private.common import DeploymentID
from ray.serve._private.config import DeploymentConfig, ReplicaConfig
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve.schema import ServeApplicationSchema
Returns the code version of an application.

    Args:
        app_config: The application config.

    Returns: a hash of the import path and (application level) runtime env representing
            the code version of the application.
    