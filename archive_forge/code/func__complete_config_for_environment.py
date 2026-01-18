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
def _complete_config_for_environment(platform_device, termination_config):
    """Complete un-filled fields of TerminationConfig based on platform."""
    if not termination_config:
        termination_config = TerminationConfig()
    if platform_device is failure_handling_util.PlatformDevice.GCE_GPU:
        return GcpGpuTerminationConfig(termination_config.termination_watcher_fn, termination_config.exit_fn, termination_config.grace_period, termination_config.save_fn)
    elif platform_device is failure_handling_util.PlatformDevice.GCE_CPU:
        return GcpCpuTerminationConfig(termination_config.termination_watcher_fn, termination_config.exit_fn, termination_config.grace_period, termination_config.save_fn)
    elif platform_device is failure_handling_util.PlatformDevice.INTERNAL_TPU:
        return BorgTPUTerminationConfig(termination_config.termination_watcher_fn, termination_config.exit_fn, termination_config.grace_period, termination_config.save_fn)
    else:
        return BorgTerminationConfig(termination_config.termination_watcher_fn, termination_config.exit_fn, termination_config.grace_period, termination_config.save_fn)