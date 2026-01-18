import collections
import errno
import functools
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import portpicker
from tensorflow.core.debug import debug_service_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.lib import grpc_debug_server
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.util import compat
def _write_graph_def(self, graph_def, device_name, wall_time):
    encoded_graph_def = graph_def.SerializeToString()
    graph_hash = int(hashlib.sha1(encoded_graph_def).hexdigest(), 16)
    event = event_pb2.Event(graph_def=encoded_graph_def, wall_time=wall_time)
    graph_file_path = os.path.join(self._dump_dir, debug_data.device_name_to_device_path(device_name), debug_data.METADATA_FILE_PREFIX + debug_data.GRAPH_FILE_TAG + debug_data.HASH_TAG + '%d_%d' % (graph_hash, wall_time))
    self._try_makedirs(os.path.dirname(graph_file_path))
    with open(graph_file_path, 'wb') as f:
        f.write(event.SerializeToString())