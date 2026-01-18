import threading
import time
from google.protobuf import text_format
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
class ShutdownAllWorkers(object):
    """Shutdown all workers.

  Processing will continue normally (typically by waiting for the down
  workers to be restarted).
  """

    def __init__(self):
        pass

    def __call__(self, run_context, all_workers, lame_workers):
        all_workers.shutdown(exit_code=42)