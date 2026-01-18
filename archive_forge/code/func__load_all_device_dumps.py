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
def _load_all_device_dumps(self, partition_graphs, validate):
    """Load the dump data for all devices."""
    device_dirs = _glob(os.path.join(self._dump_root, METADATA_FILE_PREFIX + DEVICE_TAG + '*'))
    self._device_names = []
    self._t0s = {}
    self._dump_tensor_data = {}
    self._dump_graph_file_paths = {}
    self._debug_watches = {}
    self._watch_key_to_devices = {}
    self._watch_key_to_datum = {}
    self._watch_key_to_rel_time = {}
    self._watch_key_to_dump_size_bytes = {}
    for device_dir in device_dirs:
        device_name = device_path_to_device_name(device_dir)
        self._device_names.append(device_name)
        self._load_device_dumps(device_name, device_dir)
    self._load_partition_graphs(partition_graphs, validate)
    self._calculate_t0()
    for device_name in self._device_names:
        self._create_tensor_watch_maps(device_name)