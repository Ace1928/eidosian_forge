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
@property
def _prefix(self):
    """A common prefix for all checkpoints saved with this manager.

    For example, if `directory` (a constructor argument) were `"/tmp/tf-model"`,
    `prefix` would be `"/tmp/tf-model/ckpt"` and checkpoints would generally be
    numbered `"/tmp/tf-model/ckpt-1"`, `"/tmp/tf-model/ckpt-2"`, and so on. Each
    checkpoint has several associated files
    (e.g. `"/tmp/tf-model/ckpt-2.index"`).

    Returns:
      A string prefix.
    """
    return self._checkpoint_prefix