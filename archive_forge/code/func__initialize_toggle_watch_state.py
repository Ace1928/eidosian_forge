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
def _initialize_toggle_watch_state(self, toggle_watches):
    self._toggle_watches = toggle_watches
    self._toggle_watch_state = {}
    if self._toggle_watches:
        for watch_key in self._toggle_watches:
            self._toggle_watch_state[watch_key] = False