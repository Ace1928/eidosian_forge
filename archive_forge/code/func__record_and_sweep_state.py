import collections
import os.path
import re
import time
from google.protobuf import text_format
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training_util
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _record_and_sweep_state(save_path):
    timestamp = time.time()
    if save_path in self._maybe_delete:
        del self._maybe_delete[save_path]
    self._maybe_delete[save_path] = timestamp
    self._latest_checkpoint = save_path
    self._record_state()
    self._sweep()
    self._record_state()