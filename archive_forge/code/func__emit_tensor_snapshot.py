import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
def _emit_tensor_snapshot(self, tensor, timestamp, pid, tid, value):
    """Generate Chrome Trace snapshot event for a computed Tensor.

    Args:
      tensor: A 'TensorTracker' object.
      timestamp:  The timestamp of this snapshot as a long integer.
      pid: The pid assigned for showing the device where this op ran.
      tid: The tid of the thread computing the tensor snapshot.
      value: A JSON-compliant snapshot of the object.
    """
    desc = str(value.tensor_description).replace('"', '')
    snapshot = {'tensor_description': desc}
    self._chrome_trace.emit_obj_snapshot('Tensor', tensor.name, timestamp, pid, tid, tensor.object_id, snapshot)