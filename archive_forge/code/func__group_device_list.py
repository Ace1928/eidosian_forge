import copy
from tensorflow.python import tf2
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import mirrored_run
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _group_device_list(devices):
    """Groups the devices list by task_type and task_id.

  Args:
    devices: a list of device strings for remote devices.

  Returns:
    a dict of list of device strings mapping from task_type to a list of devices
    for the task_type in the ascending order of task_id.
  """
    assert not _is_device_list_single_worker(devices)
    device_dict = {}
    for d in devices:
        d_spec = tf_device.DeviceSpec.from_string(d)
        if d_spec.job not in device_dict:
            device_dict[d_spec.job] = []
        while len(device_dict[d_spec.job]) <= d_spec.task:
            device_dict[d_spec.job].append([])
        device_dict[d_spec.job][d_spec.task].append(d)
    return device_dict