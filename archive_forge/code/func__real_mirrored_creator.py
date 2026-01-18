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
def _real_mirrored_creator(**kwargs):
    value_list = []
    for i, d in enumerate(devices):
        with ops.device(d):
            kwargs['initial_value'] = self._get_variable_creator_initial_value(replica_id=i, device=d, primary_var=value_list[0] if value_list else None, **kwargs)
            if i > 0:
                var0name = value_list[0].name.split(':')[0]
                kwargs['name'] = '%s/replica_%d/' % (var0name, i)
            with context.device_policy(context.DEVICE_PLACEMENT_SILENT):
                with record.stop_recording():
                    v = next_creator(**kwargs)
            assert not isinstance(v, values.DistributedVariable)
            value_list.append(v)
    return value_list