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
def _object_ref_deserializer(binary, call_site, owner_address, object_status):
    obj_ref = ray.ObjectRef(binary, owner_address, call_site)
    if owner_address:
        worker = ray._private.worker.global_worker
        worker.check_connected()
        context = worker.get_serialization_context()
        outer_id = context.get_outer_object_ref()
        if outer_id is None:
            outer_id = ray.ObjectRef.nil()
        worker.core_worker.deserialize_and_register_object_ref(obj_ref.binary(), outer_id, owner_address, object_status)
    return obj_ref