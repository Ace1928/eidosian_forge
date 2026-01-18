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
def _initialize_logical_devices(self):
    """Helper to initialize devices."""
    logical_devices = []
    context_devices = []
    device_list = pywrap_tfe.TFE_ContextListDevices(self._context_handle)
    try:
        self._num_gpus = 0
        current_job, current_task = (None, None)
        server_def = self._server_def or self._collective_ops_server_def
        if server_def is not None:
            current_job, current_task = (server_def.job_name, server_def.task_index)
        for i in range(pywrap_tfe.TF_DeviceListCount(device_list)):
            dev_name = pywrap_tfe.TF_DeviceListName(device_list, i)
            context_devices.append(pydev.canonical_name(dev_name))
            spec = pydev.DeviceSpec.from_string(dev_name)
            if spec.job == 'localhost':
                spec = spec.replace(job=None, replica=None, task=None)
            logical_devices.append(LogicalDevice(name=spec.to_string(), device_type=spec.device_type))
            dev_type = pywrap_tfe.TF_DeviceListType(device_list, i)
            if dev_type == 'GPU' and spec.job == current_job and (spec.task == current_task):
                self._num_gpus += 1
    finally:
        self._logical_devices = logical_devices
        self._context_devices = context_devices
        pywrap_tfe.TF_DeleteDeviceList(device_list)