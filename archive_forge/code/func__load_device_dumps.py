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
def _load_device_dumps(self, device_name, device_root):
    """Load `DebugTensorDatum` instances from the dump root of a given device.

    Populates a map {device_name: a list of `DebugTensorDatum`}, where the list
    is sorted by ascending timestamp.

    This sorting order reflects the order in which the TensorFlow executor
    processed the nodes of the graph. It is (one of many possible) topological
    sort of the nodes. This is useful for displaying tensors in the debugger
    frontend as well as for the use case in which the user wants to find a
    "culprit tensor", i.e., the first tensor in the graph that exhibits certain
    problematic properties, i.e., all zero values, or bad numerical values such
    as nan and inf.

    In addition, creates a map from node name to debug watches. In this Map,
    the key is the watched node name; the value is a dictionary.
    Of this dictionary, the key is the watched_output_slot.

    This method attempts to load the debug watches from the tensor dump files
    first, before loading the full set of debug watches from the partition
    graphs as done later. This is necessary because sometimes the partition
    graphs may not be available, e.g., when the run errors out.

    Args:
      device_name: (`str`) name of the device.
      device_root: (`str`) dump root directory of the given device.

    Raises:
      ValueError: If GraphDef for the device is not available.
    """
    self._dump_tensor_data[device_name] = []
    self._debug_watches[device_name] = collections.defaultdict(lambda: collections.defaultdict(set))
    for root, _, files in gfile.Walk(device_root):
        for f in files:
            if _is_graph_file(f):
                self._dump_graph_file_paths[device_name] = os.path.join(root, f)
            else:
                datum = self._dump_file_name_to_datum(root, f)
                self._dump_tensor_data[device_name].append(datum)
                self._debug_watches[device_name][datum.node_name][datum.output_slot].add(datum.debug_op)
    self._dump_tensor_data[device_name] = sorted(self._dump_tensor_data[device_name], key=lambda x: x.extended_timestamp)
    if self._dump_tensor_data[device_name]:
        self._t0s[device_name] = self._dump_tensor_data[device_name][0].timestamp
    else:
        self._t0s[device_name] = None