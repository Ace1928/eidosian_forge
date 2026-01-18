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
def _write_value_event(self, event):
    value = event.summary.value[0]
    summary_metadata = event.summary.value[0].metadata
    if not summary_metadata.plugin_data:
        raise ValueError('The value lacks plugin data.')
    try:
        content = json.loads(compat.as_text(summary_metadata.plugin_data.content))
    except ValueError as err:
        raise ValueError('Could not parse content into JSON: %r, %r' % (content, err))
    device_name = content['device']
    dump_full_path = _get_dump_file_path(self._dump_dir, device_name, value.node_name)
    self._try_makedirs(os.path.dirname(dump_full_path))
    with open(dump_full_path, 'wb') as f:
        f.write(event.SerializeToString())