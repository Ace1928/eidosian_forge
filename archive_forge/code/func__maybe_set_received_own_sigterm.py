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
def _maybe_set_received_own_sigterm(self):
    """Claim earliest preemption if no one else has done it before."""
    if self._local_mode:
        logging.info('Member %s has received termination notice.', self._id_in_cluster)
        self._received_own_sigterm_time = time.time()
        self._received_own_sigterm.set()
        return
    try:
        context.context().set_config_key_value(_PREEMPTION_WORKER_KEY, self._id_in_cluster)
        logging.info('Member %s has received termination notice.', self._id_in_cluster)
        self._received_own_sigterm_time = time.time()
        self._received_own_sigterm.set()
    except errors.AlreadyExistsError:
        logging.info('Member %s has received termination notice. But some other worker has received it as well! Leaving it to them to decide when to checkpoint. ', self._id_in_cluster)
        return