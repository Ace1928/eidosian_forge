import contextlib
import logging
import threading
from typing import Any, List, Sequence, Set
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python import _pywrap_dtensor_device
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.util import _pywrap_utils
def _create_host_array(self, shape, host_id):
    """Returns ID and device lists that can be used to create a host mesh."""
    num_global_devices = np.prod(shape)
    global_device_ids = np.arange(num_global_devices).reshape(shape)
    local_device_list = [tf_device.DeviceSpec(job=config.full_job_name(), device_type='CPU', device_index=0)]
    num_local_devices = len(local_device_list)
    local_device_ids = [x + host_id * num_local_devices for x in range(num_local_devices)]
    return (global_device_ids, local_device_ids, local_device_list)