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
def _get_dump_file_path(dump_root, device_name, debug_node_name):
    """Get the file path of the dump file for a debug node.

  Args:
    dump_root: (str) Root dump directory.
    device_name: (str) Name of the device that the debug node resides on.
    debug_node_name: (str) Name of the debug node, e.g.,
      cross_entropy/Log:0:DebugIdentity.

  Returns:
    (str) Full path of the dump file.
  """
    dump_root = os.path.join(dump_root, debug_data.device_name_to_device_path(device_name))
    if '/' in debug_node_name:
        dump_dir = os.path.join(dump_root, os.path.dirname(debug_node_name))
        dump_file_name = re.sub(':', '_', os.path.basename(debug_node_name))
    else:
        dump_dir = dump_root
        dump_file_name = re.sub(':', '_', debug_node_name)
    now_microsec = int(round(time.time() * 1000 * 1000))
    dump_file_name += '_%d' % now_microsec
    return os.path.join(dump_dir, dump_file_name)