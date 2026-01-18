import os
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.summary import summary as tf_summary
from tensorflow.python.training import saver
def _write_custom_summaries(self, step, logs=None):
    """Writes metrics out as custom scalar summaries.

    Args:
        step: the global step to use for TensorBoard.
        logs: dict. Keys are scalar summary names, values are
            NumPy scalars.

    """
    logs = logs or {}
    if context.executing_eagerly():
        with self.writer.as_default(), summary_ops_v2.record_if(True):
            for name, value in logs.items():
                if isinstance(value, np.ndarray):
                    value = value.item()
                summary_ops_v2.scalar(name, value, step=step)
    else:
        for name, value in logs.items():
            if isinstance(value, np.ndarray):
                value = value.item()
            summary = tf_summary.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, step)
    self.writer.flush()