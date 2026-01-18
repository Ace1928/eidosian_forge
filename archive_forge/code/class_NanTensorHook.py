import os
import time
import numpy as np
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.client import timeline
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.NanTensorHook'])
class NanTensorHook(session_run_hook.SessionRunHook):
    """Monitors the loss tensor and stops training if loss is NaN.

  Can either fail with exception or just stop training.
  """

    def __init__(self, loss_tensor, fail_on_nan_loss=True):
        """Initializes a `NanTensorHook`.

    Args:
      loss_tensor: `Tensor`, the loss tensor.
      fail_on_nan_loss: `bool`, whether to raise exception when loss is NaN.
    """
        self._loss_tensor = loss_tensor
        self._fail_on_nan_loss = fail_on_nan_loss

    def before_run(self, run_context):
        return SessionRunArgs(self._loss_tensor)

    def after_run(self, run_context, run_values):
        if np.isnan(run_values.results):
            failure_message = 'Model diverged with loss = NaN.'
            if self._fail_on_nan_loss:
                logging.error(failure_message)
                raise NanLossDuringTrainingError
            else:
                logging.warning(failure_message)
                run_context.request_stop()