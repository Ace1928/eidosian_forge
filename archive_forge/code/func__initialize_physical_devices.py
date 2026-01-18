import collections
import contextlib
import copy
import gc
import itertools
import os
import random
import threading
from absl import logging
import numpy as np
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import execute
from tensorflow.python.eager import executor
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import tfrt_utils
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tsl.protobuf import coordination_config_pb2
def _initialize_physical_devices(self, reinitialize=False):
    """Gets local devices visible to the system.

    Args:
      reinitialize: If True, reinitializes self._physical_devices  so that
        dynamic registered devices will also be visible to the python front-end.
    """
    with self._device_lock:
        if not reinitialize and self._physical_devices is not None:
            return
        devs = pywrap_tfe.TF_ListPhysicalDevices()
        self._physical_devices = [PhysicalDevice(name=d.decode(), device_type=d.decode().split(':')[1]) for d in devs]
        self._physical_device_to_index = {p: i for i, p in enumerate(self._physical_devices)}
        pluggable_devs = pywrap_tfe.TF_ListPluggablePhysicalDevices()
        self._pluggable_devices = [PhysicalDevice(name=d.decode(), device_type=d.decode().split(':')[1]) for d in pluggable_devs]
        self._visible_device_list = list(self._physical_devices)
        self._memory_growth_map = {d: None for d in self._physical_devices if d.device_type == 'GPU' or d in self._pluggable_devices}
    self._import_config()