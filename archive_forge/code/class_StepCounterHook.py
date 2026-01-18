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
@tf_export(v1=['train.StepCounterHook'])
class StepCounterHook(session_run_hook.SessionRunHook):
    """Hook that counts steps per second."""

    def __init__(self, every_n_steps=100, every_n_secs=None, output_dir=None, summary_writer=None):
        if (every_n_steps is None) == (every_n_secs is None):
            raise ValueError('exactly one of every_n_steps and every_n_secs should be provided.')
        self._timer = SecondOrStepTimer(every_steps=every_n_steps, every_secs=every_n_secs)
        self._summary_writer = summary_writer
        self._output_dir = output_dir
        self._last_global_step = None
        self._steps_per_run = 1

    def _set_steps_per_run(self, steps_per_run):
        self._steps_per_run = steps_per_run

    def begin(self):
        if self._summary_writer is None and self._output_dir:
            self._summary_writer = SummaryWriterCache.get(self._output_dir)
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        if self._global_step_tensor is None:
            raise RuntimeError('Global step should be created to use StepCounterHook.')
        self._summary_tag = training_util.get_global_step().op.name + '/sec'

    def before_run(self, run_context):
        return SessionRunArgs(self._global_step_tensor)

    def _log_and_record(self, elapsed_steps, elapsed_time, global_step):
        steps_per_sec = elapsed_steps / elapsed_time
        if self._summary_writer is not None:
            summary = Summary(value=[Summary.Value(tag=self._summary_tag, simple_value=steps_per_sec)])
            self._summary_writer.add_summary(summary, global_step)
        logging.info('%s: %g', self._summary_tag, steps_per_sec)

    def after_run(self, run_context, run_values):
        _ = run_context
        stale_global_step = run_values.results
        if self._timer.should_trigger_for_step(stale_global_step + self._steps_per_run):
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(global_step)
                if elapsed_time is not None:
                    self._log_and_record(elapsed_steps, elapsed_time, global_step)
        if stale_global_step == self._last_global_step:
            logging.log_first_n(logging.WARN, 'It seems that global step (tf.train.get_global_step) has not been increased. Current value (could be stable): %s vs previous value: %s. You could increase the global step by passing tf.train.get_global_step() to Optimizer.apply_gradients or Optimizer.minimize.', 5, stale_global_step, self._last_global_step)
        self._last_global_step = stale_global_step