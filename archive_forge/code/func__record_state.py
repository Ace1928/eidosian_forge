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
def _record_state(self):
    """Saves the `CheckpointManager`'s state in `directory`."""
    filenames, timestamps = zip(*self._maybe_delete.items())
    update_checkpoint_state_internal(self._directory, model_checkpoint_path=self.latest_checkpoint, all_model_checkpoint_paths=filenames, all_model_checkpoint_timestamps=timestamps, last_preserved_timestamp=self._last_preserved_timestamp, save_relative_paths=True)