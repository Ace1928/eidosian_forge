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
def _maybe_create_checkpoint_manager(self):
    """Create CheckpointManager(s) if a checkpoint is passed else take it."""
    if isinstance(self._checkpoint_or_checkpoint_manager, checkpoint_management.CheckpointManager):
        self._read_checkpoint_manager = self._checkpoint_or_checkpoint_manager
        self._write_checkpoint_manager = self._checkpoint_or_checkpoint_manager
        self._api_made_checkpoint_manager = False
    else:
        self._api_made_checkpoint_manager = True
        self._read_checkpoint_manager = checkpoint_management.CheckpointManager(self._checkpoint_or_checkpoint_manager, directory=self._checkpoint_dir, max_to_keep=1)
        if self._is_chief:
            self._write_checkpoint_manager = self._read_checkpoint_manager
        else:
            self._write_checkpoint_manager = checkpoint_management.CheckpointManager(self._checkpoint_or_checkpoint_manager, _non_chief_checkpoint_dir(self._checkpoint_dir, self._cluster_resolver.task_id), max_to_keep=1)