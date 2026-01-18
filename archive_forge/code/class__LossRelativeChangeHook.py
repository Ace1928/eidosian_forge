from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clustering_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
class _LossRelativeChangeHook(tf.compat.v1.train.SessionRunHook):
    """Stops when the change in loss goes below a tolerance."""

    def __init__(self, loss_tensor, tolerance):
        """Creates a _LossRelativeChangeHook.

    Args:
      loss_tensor: A scalar tensor of the loss value.
      tolerance: A relative tolerance of loss change between iterations.
    """
        self._loss_tensor = loss_tensor
        self._tolerance = tolerance
        self._prev_loss = None

    def before_run(self, run_context):
        del run_context
        return tf.compat.v1.train.SessionRunArgs(self._loss_tensor)

    def after_run(self, run_context, run_values):
        loss = run_values.results
        assert loss is not None
        if self._prev_loss:
            relative_change = abs(loss - self._prev_loss) / (1 + abs(self._prev_loss))
            if relative_change < self._tolerance:
                run_context.request_stop()
        self._prev_loss = loss