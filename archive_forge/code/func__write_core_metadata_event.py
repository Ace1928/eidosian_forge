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
def _write_core_metadata_event(self, event):
    core_metadata_path = os.path.join(self._dump_dir, debug_data.METADATA_FILE_PREFIX + debug_data.CORE_METADATA_TAG + '_%d' % event.wall_time)
    self._try_makedirs(self._dump_dir)
    with open(core_metadata_path, 'wb') as f:
        f.write(event.SerializeToString())