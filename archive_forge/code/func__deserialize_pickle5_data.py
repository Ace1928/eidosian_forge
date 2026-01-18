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
def _deserialize_pickle5_data(self, data):
    try:
        in_band, buffers = unpack_pickle5_buffers(data)
        if len(buffers) > 0:
            obj = pickle.loads(in_band, buffers=buffers)
        else:
            obj = pickle.loads(in_band)
    except pickle.pickle.PicklingError:
        raise DeserializationError()
    return obj