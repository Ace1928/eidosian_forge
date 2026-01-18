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
def _infer_device_name(self, device_name, node_name):
    """Infer the device name given node name.

    If device_name is provided (i.e., not None), it'll be simply returned right
    away.

    Args:
      device_name: (str or None) name of the device. If None, will try to infer
        the device name by looking at the available nodes.
      node_name: (str) name of the node.

    Returns:
      (str) Inferred name of the device, if available.

    Raises:
      ValueError: If the node name does not exist on any of the available
        devices or if there are multiple devices that contain the node with
        the given name.
    """
    if device_name is None:
        if node_name in self._node_devices:
            if len(self._node_devices[node_name]) == 1:
                return list(self._node_devices[node_name])[0]
            else:
                raise ValueError("There are multiple (%d) devices with nodes named '%s' but device_name is not specified." % (len(self._node_devices[node_name]), node_name))
        else:
            raise ValueError("None of the %d device(s) has a node named '%s'." % (len(self._device_names), node_name))
    else:
        return device_name