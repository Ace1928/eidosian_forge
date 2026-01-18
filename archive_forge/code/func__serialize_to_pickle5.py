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
def _serialize_to_pickle5(self, metadata, value):
    writer = Pickle5Writer()
    try:
        self.set_in_band_serialization()
        inband = pickle.dumps(value, protocol=5, buffer_callback=writer.buffer_callback)
    except Exception as e:
        self.get_and_clear_contained_object_refs()
        raise e
    finally:
        self.set_out_of_band_serialization()
    return Pickle5SerializedObject(metadata, inband, writer, self.get_and_clear_contained_object_refs())