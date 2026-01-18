from typing import List, Optional, Tuple
from absl import logging
import numpy as np
from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python import tpu_util
from tensorflow.python.eager import context
from tensorflow.python.framework import device as tf_device
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export
def _make_device_specs(devices: Optional[List[str]]=None, device_type: Optional[str]=None) -> Tuple[List[tf_device.DeviceSpec], str]:
    """Makes device specs from local devices names or number of global devices."""
    if devices is None:
        if device_type is None:
            device_type = 'CPU'
        devices = config.local_devices(device_type)
    else:
        devices = [tf_device.DeviceSpec.from_string(d) for d in devices]
        if device_type is None:
            device_type = devices[0].device_type
        if device_type.upper() != devices[0].device_type.upper():
            raise ValueError(f'Conflicting devices {str(devices)} and device_type {device_type}')
    return (devices, device_type)