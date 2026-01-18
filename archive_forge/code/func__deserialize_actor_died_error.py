import io
import logging
import threading
import traceback
from typing import Any
import google.protobuf.message
import ray._private.utils
import ray.cloudpickle as pickle
from ray._private import ray_constants
from ray._raylet import (
from ray.core.generated.common_pb2 import ErrorType, RayErrorInfo
from ray.exceptions import (
from ray.util import serialization_addons
from ray.util import inspect_serializability
def _deserialize_actor_died_error(self, data, metadata_fields):
    if not data:
        return RayActorError()
    ray_error_info = self._deserialize_error_info(data, metadata_fields)
    assert ray_error_info.HasField('actor_died_error')
    if ray_error_info.actor_died_error.HasField('creation_task_failure_context'):
        return RayError.from_ray_exception(ray_error_info.actor_died_error.creation_task_failure_context)
    else:
        assert ray_error_info.actor_died_error.HasField('actor_died_error_context')
        return RayActorError(cause=ray_error_info.actor_died_error.actor_died_error_context)