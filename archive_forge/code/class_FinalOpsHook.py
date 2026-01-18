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
@tf_export(v1=['train.FinalOpsHook'])
class FinalOpsHook(session_run_hook.SessionRunHook):
    """A hook which evaluates `Tensors` at the end of a session."""

    def __init__(self, final_ops, final_ops_feed_dict=None):
        """Initializes `FinalOpHook` with ops to run at the end of the session.

    Args:
      final_ops: A single `Tensor`, a list of `Tensors` or a dictionary of names
        to `Tensors`.
      final_ops_feed_dict: A feed dictionary to use when running
        `final_ops_dict`.
    """
        self._final_ops = final_ops
        self._final_ops_feed_dict = final_ops_feed_dict
        self._final_ops_values = None

    @property
    def final_ops_values(self):
        return self._final_ops_values

    def end(self, session):
        if self._final_ops is not None:
            try:
                self._final_ops_values = session.run(self._final_ops, feed_dict=self._final_ops_feed_dict)
            except (errors.OutOfRangeError, StopIteration) as e:
                logging.warning('An OutOfRangeError or StopIteration exception is raised by the code in FinalOpsHook. This typically means the Ops running by the FinalOpsHook have a dependency back to some input source, which should not happen. For example, for metrics in tf.estimator.Estimator, all metrics functions return two Ops: `value_op` and  `update_op`. Estimator.evaluate calls the `update_op` for each batch of the data in input source and, once it is exhausted, it call the `value_op` to get the metric values. The `value_op` here should have dependency back to variables reading only, rather than reading another batch from input. Otherwise, the `value_op`, executed by `FinalOpsHook`, triggers another data reading, which ends OutOfRangeError/StopIteration. Please fix that.')
                raise e