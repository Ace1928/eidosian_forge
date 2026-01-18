import contextvars
import logging
from dataclasses import dataclass
from typing import Callable, Optional
import ray
from ray.exceptions import RayActorError
from ray.serve._private.client import ServeControllerClient
from ray.serve._private.common import ReplicaTag
from ray.serve._private.constants import SERVE_CONTROLLER_NAME, SERVE_NAMESPACE
from ray.serve.exceptions import RayServeException
from ray.serve.grpc_util import RayServegRPCContext
from ray.util.annotations import DeveloperAPI
def _get_internal_replica_context():
    return _INTERNAL_REPLICA_CONTEXT