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
@tf_export(v1=['train.StopAtStepHook'])
class StopAtStepHook(session_run_hook.SessionRunHook):
    """Hook that requests stop at a specified step.

  @compatibility(TF2)
  Please check this [notebook][notebook] on how to migrate the API to TF2.

  [notebook]:https://github.com/tensorflow/docs/blob/master/site/en/guide/migrate/logging_stop_hook.ipynb

  @end_compatibility
  """

    def __init__(self, num_steps=None, last_step=None):
        """Initializes a `StopAtStepHook`.

    This hook requests stop after either a number of steps have been
    executed or a last step has been reached. Only one of the two options can be
    specified.

    if `num_steps` is specified, it indicates the number of steps to execute
    after `begin()` is called. If instead `last_step` is specified, it
    indicates the last step we want to execute, as passed to the `after_run()`
    call.

    Args:
      num_steps: Number of steps to execute.
      last_step: Step after which to stop.

    Raises:
      ValueError: If one of the arguments is invalid.
    """
        if num_steps is None and last_step is None:
            raise ValueError('One of num_steps or last_step must be specified.')
        if num_steps is not None and last_step is not None:
            raise ValueError('Only one of num_steps or last_step can be specified.')
        self._num_steps = num_steps
        self._last_step = last_step

    def begin(self):
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        if self._global_step_tensor is None:
            raise RuntimeError('Global step should be created to use StopAtStepHook.')

    def after_create_session(self, session, coord):
        if self._last_step is None:
            global_step = session.run(self._global_step_tensor)
            self._last_step = global_step + self._num_steps

    def before_run(self, run_context):
        return SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        global_step = run_values.results + 1
        if global_step >= self._last_step:
            step = run_context.session.run(self._global_step_tensor)
            if step >= self._last_step:
                run_context.request_stop()