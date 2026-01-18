from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
class _OneDeviceReplicaContext(distribute_lib.ReplicaContext):
    """ReplicaContext for OneDeviceStrategy."""

    def __init__(self, strategy):
        distribute_lib.ReplicaContext.__init__(self, strategy, replica_id_in_sync_group=0)

    @property
    def devices(self):
        return self._strategy.extended.worker_devices