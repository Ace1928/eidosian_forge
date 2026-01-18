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
def _is_device_list_single_worker(devices):
    """Checks whether the devices list is for single or multi-worker.

  Args:
    devices: a list of device strings or tf.config.LogicalDevice objects, for
      either local or for remote devices.

  Returns:
    a boolean indicating whether these device strings are for local or for
    remote.

  Raises:
    ValueError: if device strings are not consistent.
  """
    specs = []
    for d in devices:
        name = d.name if isinstance(d, context.LogicalDevice) else d
        specs.append(tf_device.DeviceSpec.from_string(name))
    num_workers = len({(d.job, d.task, d.replica) for d in specs})
    all_local = all((d.job in (None, 'localhost') for d in specs))
    any_local = any((d.job in (None, 'localhost') for d in specs))
    if any_local and (not all_local):
        raise ValueError("Local device should have only 'localhost' in the job field in device string. E.g. 'job:localhost' in /job:localhost/replica:0/task:0/device:CPU:0Devices cannot have mixed list of device strings containing both localhost and other job types such as worker, ps etc. ")
    if num_workers == 1 and (not all_local):
        if any((d.task is None for d in specs)):
            raise ValueError("Remote device string must have task specified.E.g. 'task:0' in /job:worker/replica:0/task:0/device:CPU:0")
    return num_workers == 1