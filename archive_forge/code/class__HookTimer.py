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
class _HookTimer:
    """Base timer for determining when Hooks should trigger.

  Should not be instantiated directly.
  """

    def __init__(self):
        pass

    def reset(self):
        """Resets the timer."""
        pass

    def should_trigger_for_step(self, step):
        """Return true if the timer should trigger for the specified step."""
        raise NotImplementedError

    def update_last_triggered_step(self, step):
        """Update the last triggered time and step number.

    Args:
      step: The current step.

    Returns:
      A pair `(elapsed_time, elapsed_steps)`, where `elapsed_time` is the number
      of seconds between the current trigger and the last one (a float), and
      `elapsed_steps` is the number of steps between the current trigger and
      the last one. Both values will be set to `None` on the first trigger.
    """
        raise NotImplementedError

    def last_triggered_step(self):
        """Returns the last triggered time step or None if never triggered."""
        raise NotImplementedError