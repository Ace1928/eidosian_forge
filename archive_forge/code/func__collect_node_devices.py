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
def _collect_node_devices(self, debug_graph):
    for node_name in debug_graph.node_devices:
        if node_name in self._node_devices:
            self._node_devices[node_name] = self._node_devices[node_name].union(debug_graph.node_devices[node_name])
        else:
            self._node_devices[node_name] = debug_graph.node_devices[node_name]