import collections
import glob
import json
import os
import platform
import re
import numpy as np
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
def _satisfied_at_timestamp(self, device_name, pending, timestamp, start_i=0):
    """Determine whether pending inputs are satisfied at given timestamp.

    Note: This method mutates the input argument "pending".

    Args:
      device_name: (str) device name.
      pending: A list of 2-tuple (node_name, output_slot): the dependencies to
        check.
      timestamp: (int) the timestamp in question.
      start_i: (int) the index in self._dump_tensor_data to start searching for
        the timestamp.

    Returns:
      (bool) Whether all the dependencies in pending are satisfied at the
        timestamp. If pending is empty to begin with, return True.
    """
    if not pending:
        return True
    for datum in self._dump_tensor_data[device_name][start_i:]:
        if datum.timestamp > timestamp:
            break
        if datum.timestamp == timestamp and (datum.node_name, datum.output_slot) in pending:
            pending.remove((datum.node_name, datum.output_slot))
            if not pending:
                return True
    return not pending