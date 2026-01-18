import threading
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.options import ExternalStatePolicy
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import type_spec as type_spec_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@tf_export('distribute.experimental.coordinator.PerWorkerValues', 'distribute.coordinator.PerWorkerValue', v1=[])
class PerWorkerValues(composite_tensor.CompositeTensor):
    """A container that holds a list of values, one value per worker.

  `tf.distribute.experimental.coordinator.PerWorkerValues` contains a collection
  of values, where each of the values is located on its corresponding worker,
  and upon being used as one of the `args` or `kwargs` of
  `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule()`, the
  value specific to a worker will be passed into the function being executed at
  that corresponding worker.

  Currently, the only supported path to create an object of
  `tf.distribute.experimental.coordinator.PerWorkerValues` is through calling
  `iter` on a `ClusterCoordinator.create_per_worker_dataset`-returned
  distributed dataset instance. The mechanism to create a custom
  `tf.distribute.experimental.coordinator.PerWorkerValues` is not yet supported.
  """

    def __init__(self, values):
        for v in values:
            if not isinstance(v, remote_value.RemoteValue):
                raise AssertionError('`PerWorkerValues` should only take `RemoteValue`s.')
        self._values = tuple(values)

    @property
    def _type_spec(self):
        return PerWorkerValuesTypeSpec(self._values[0]._type_spec, type(self))