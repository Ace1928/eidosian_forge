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
def _set_internal_replica_context(*, app_name: str, deployment: str, replica_tag: ReplicaTag, servable_object: Callable, controller_name: str):
    global _INTERNAL_REPLICA_CONTEXT
    _INTERNAL_REPLICA_CONTEXT = ReplicaContext(app_name=app_name, deployment=deployment, replica_tag=replica_tag, servable_object=servable_object, _internal_controller_name=controller_name)