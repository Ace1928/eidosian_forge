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
def _load_core_metadata(self):
    core_metadata_files = _glob(os.path.join(self._dump_root, METADATA_FILE_PREFIX + CORE_METADATA_TAG + '*'))
    for core_metadata_file in core_metadata_files:
        with gfile.Open(core_metadata_file, 'rb') as f:
            event = event_pb2.Event()
            event.ParseFromString(f.read())
            self._core_metadata.append(extract_core_metadata_from_event_proto(event))