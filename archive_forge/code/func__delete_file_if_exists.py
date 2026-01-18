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
def _delete_file_if_exists(filespec):
    """Deletes files matching `filespec`."""
    for pathname in file_io.get_matching_files(filespec):
        try:
            file_io.delete_file(pathname)
        except errors.NotFoundError:
            logging.warning("Hit NotFoundError when deleting '%s', possibly because another process/thread is also deleting/moving the same file", pathname)