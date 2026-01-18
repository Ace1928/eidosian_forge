from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
from tensorflow.python.util.tf_export import tf_export
Choose a device for `op`.

    Args:
      op: an `Operation`.

    Returns:
      The device to use for the `Operation`.
    