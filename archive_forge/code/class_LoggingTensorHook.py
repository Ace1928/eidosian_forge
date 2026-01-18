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
@tf_export(v1=['train.LoggingTensorHook'])
class LoggingTensorHook(session_run_hook.SessionRunHook):
    """Prints the given tensors every N local steps, every N seconds, or at end.

  The tensors will be printed to the log, with `INFO` severity. If you are not
  seeing the logs, you might want to add the following line after your imports:

  ```python
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  ```

  Note that if `at_end` is True, `tensors` should not include any tensor
  whose evaluation produces a side effect such as consuming additional inputs.

  @compatibility(TF2)
  Please check this [notebook][notebook] on how to migrate the API to TF2.

  [notebook]:https://github.com/tensorflow/docs/blob/master/site/en/guide/migrate/logging_stop_hook.ipynb

  @end_compatibility

  """

    def __init__(self, tensors, every_n_iter=None, every_n_secs=None, at_end=False, formatter=None):
        """Initializes a `LoggingTensorHook`.

    Args:
      tensors: `dict` that maps string-valued tags to tensors/tensor names, or
        `iterable` of tensors/tensor names.
      every_n_iter: `int`, print the values of `tensors` once every N local
        steps taken on the current worker.
      every_n_secs: `int` or `float`, print the values of `tensors` once every N
        seconds. Exactly one of `every_n_iter` and `every_n_secs` should be
        provided.
      at_end: `bool` specifying whether to print the values of `tensors` at the
        end of the run.
      formatter: function, takes dict of `tag`->`Tensor` and returns a string.
        If `None` uses default printing all tensors.

    Raises:
      ValueError: if `every_n_iter` is non-positive.
    """
        only_log_at_end = at_end and every_n_iter is None and (every_n_secs is None)
        if not only_log_at_end and (every_n_iter is None) == (every_n_secs is None):
            raise ValueError('either at_end and/or exactly one of every_n_iter and every_n_secs must be provided.')
        if every_n_iter is not None and every_n_iter <= 0:
            raise ValueError('invalid every_n_iter=%s.' % every_n_iter)
        if not isinstance(tensors, dict):
            self._tag_order = tensors
            tensors = {item: item for item in tensors}
        else:
            self._tag_order = sorted(tensors.keys())
        self._tensors = tensors
        self._formatter = formatter
        self._timer = NeverTriggerTimer() if only_log_at_end else SecondOrStepTimer(every_secs=every_n_secs, every_steps=every_n_iter)
        self._log_at_end = at_end

    def begin(self):
        self._timer.reset()
        self._iter_count = 0
        self._current_tensors = {tag: _as_graph_element(tensor) for tag, tensor in self._tensors.items()}

    def before_run(self, run_context):
        self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
        if self._should_trigger:
            return SessionRunArgs(self._current_tensors)
        else:
            return None

    def _log_tensors(self, tensor_values):
        original = np.get_printoptions()
        np.set_printoptions(suppress=True)
        elapsed_secs, _ = self._timer.update_last_triggered_step(self._iter_count)
        if self._formatter:
            logging.info(self._formatter(tensor_values))
        else:
            stats = []
            for tag in self._tag_order:
                stats.append('%s = %s' % (tag, tensor_values[tag]))
            if elapsed_secs is not None:
                logging.info('%s (%.3f sec)', ', '.join(stats), elapsed_secs)
            else:
                logging.info('%s', ', '.join(stats))
        np.set_printoptions(**original)

    def after_run(self, run_context, run_values):
        _ = run_context
        if self._should_trigger:
            self._log_tensors(run_values.results)
        self._iter_count += 1

    def end(self, session):
        if self._log_at_end:
            values = session.run(self._current_tensors)
            self._log_tensors(values)