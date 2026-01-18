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
class _OneCoreTPUContext(_InternalTPUContext):
    """Special _InternalTPUContext for one core usage."""

    def __init__(self, config, train_batch_size, eval_batch_size, predict_batch_size, use_tpu):
        super(_OneCoreTPUContext, self).__init__(config, train_batch_size, eval_batch_size, predict_batch_size, use_tpu)

    def _get_tpu_system_metadata(self):
        """Gets the (maybe cached) TPU system metadata."""
        master = self._get_master_address()
        tpu_system_metadata = self._lazy_tpu_system_metadata_dict.get(master)
        if tpu_system_metadata is not None:
            return tpu_system_metadata
        tpu_system_metadata = tf.tpu.experimental.TPUSystemMetadata(num_cores=1, num_hosts=1, num_of_cores_per_host=1, topology=None, devices=[])
        self._lazy_tpu_system_metadata_dict[master] = tpu_system_metadata
        return tpu_system_metadata