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
def _create_tensor_watch_maps(self, device_name):
    """Create maps from tensor watch keys to datum and to timestamps.

    Create a map from watch key (tensor name + debug op) to `DebugTensorDatum`
    item. Also make a map from watch key to relative timestamp.
    "relative" means (absolute timestamp - t0).

    Args:
      device_name: (str) name of the device.
    """
    self._watch_key_to_datum[device_name] = {}
    self._watch_key_to_rel_time[device_name] = {}
    self._watch_key_to_dump_size_bytes[device_name] = {}
    for datum in self._dump_tensor_data[device_name]:
        if datum.watch_key not in self._watch_key_to_devices:
            self._watch_key_to_devices[datum.watch_key] = {device_name}
        else:
            self._watch_key_to_devices[datum.watch_key].add(device_name)
        if datum.watch_key not in self._watch_key_to_datum[device_name]:
            self._watch_key_to_datum[device_name][datum.watch_key] = [datum]
            self._watch_key_to_rel_time[device_name][datum.watch_key] = [datum.timestamp - self._t0]
            self._watch_key_to_dump_size_bytes[device_name][datum.watch_key] = [datum.dump_size_bytes]
        else:
            self._watch_key_to_datum[device_name][datum.watch_key].append(datum)
            self._watch_key_to_rel_time[device_name][datum.watch_key].append(datum.timestamp - self._t0)
            self._watch_key_to_dump_size_bytes[device_name][datum.watch_key].append(datum.dump_size_bytes)