import contextlib
import os
import time
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary as _summary
from tensorflow.python.training import coordinator
from tensorflow.python.training import saver as saver_mod
from tensorflow.python.training import session_manager as session_manager_mod
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _get_first_op_from_collection(self, key):
    """Returns the first `Operation` from a collection.

    Args:
      key: A string collection key.

    Returns:
      The first Op found in a collection, or `None` if the collection is empty.
    """
    try:
        op_list = ops.get_collection(key)
        if len(op_list) > 1:
            logging.info('Found %d %s operations. Returning the first one.', len(op_list), key)
        if op_list:
            return op_list[0]
    except LookupError:
        pass
    return None