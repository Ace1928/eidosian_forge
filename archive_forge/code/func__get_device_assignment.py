from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from contextlib import contextmanager
import copy
import tensorflow as tf
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.tpu import device_assignment as tpu_device_assignment
from tensorflow.python.tpu import tpu_system_metadata as tpu_system_metadata_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.tpu import _tpu_estimator_embedding
from tensorflow_estimator.python.estimator.tpu import tpu_config
def _get_device_assignment(self):
    """Gets the (maybe cached) TPU device assignment."""
    master = self._get_master_address()
    device_assignment = self._lazy_device_assignment_dict.get(master)
    if device_assignment is not None:
        return device_assignment
    tpu_system_metadata = self._get_tpu_system_metadata()
    device_assignment = tpu_device_assignment.device_assignment(tpu_system_metadata.topology, computation_shape=self._computation_shape, num_replicas=self.num_replicas)
    tf.compat.v1.logging.info('num_cores_per_replica: %s', str(self._config.tpu_config.num_cores_per_replica))
    tf.compat.v1.logging.info('computation_shape: %s', str(self._computation_shape))
    tf.compat.v1.logging.info('num_replicas: %d', self.num_replicas)
    tf.compat.v1.logging.info('device_assignment.topology.device_coordinates: %s', str(device_assignment.topology.device_coordinates))
    tf.compat.v1.logging.info('device_assignment.core_assignment: %s', str(device_assignment.core_assignment))
    self._lazy_device_assignment_dict[master] = device_assignment
    return device_assignment