import copy
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import mirrored_run
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import device_setter
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _verify_destinations_not_different_worker(self, destinations):
    if not self._cluster_spec:
        return
    if destinations is None:
        return
    for d in cross_device_ops_lib.get_devices_from(destinations):
        d_spec = tf_device.DeviceSpec.from_string(d)
        if d_spec.job == self._task_type and d_spec.task != self._task_id:
            raise ValueError('Cannot reduce to another worker: %r, current worker is %r' % (d, self._worker_device))