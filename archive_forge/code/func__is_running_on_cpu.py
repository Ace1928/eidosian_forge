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
def _is_running_on_cpu(self, is_export_mode):
    """Determines whether the input_fn and model_fn should be invoked on CPU."""
    mode = self._assert_mode()
    if not self._use_tpu:
        return True
    if mode == model_fn_lib.ModeKeys.EVAL and (not self._eval_on_tpu):
        tf.compat.v1.logging.info('_is_running_on_cpu: eval_on_tpu disabled')
        return True
    if is_export_mode:
        return True
    return False