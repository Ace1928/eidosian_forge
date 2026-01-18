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
def add_contained_object_ref(self, object_ref):
    if self.is_in_band_serialization():
        if not hasattr(self._thread_local, 'object_refs'):
            self._thread_local.object_refs = set()
        self._thread_local.object_refs.add(object_ref)
    else:
        ray._private.worker.global_worker.core_worker.add_object_ref_reference(object_ref)