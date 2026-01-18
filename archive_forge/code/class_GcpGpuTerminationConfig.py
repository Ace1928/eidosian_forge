import os
import signal
import sys
import threading
import time
from tensorflow.core.distributed_runtime.preemption import gen_check_preemption_op
from tensorflow.python.checkpoint import checkpoint as checkpoint_lib
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute.failure_handling import failure_handling_util
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
class GcpGpuTerminationConfig(TerminationConfig):
    """Configurations for GCP GPU VM."""

    def __init__(self, termination_watcher_fn=None, exit_fn=None, grace_period=None, save_fn=None):
        self.termination_watcher_fn = termination_watcher_fn or failure_handling_util.termination_watcher_function_gce
        self.exit_fn = exit_fn or failure_handling_util.gce_exit_fn
        self.grace_period = grace_period if grace_period or grace_period == 0 else failure_handling_util.GRACE_PERIOD_GCE
        self.save_fn = save_fn