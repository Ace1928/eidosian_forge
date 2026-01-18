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
def _load_feeds_info(self):
    feeds_info_files = _glob(os.path.join(self._dump_root, METADATA_FILE_PREFIX + FEED_KEYS_INFO_FILE_TAG + '*'))
    self._run_feed_keys_info = []
    for feeds_info_file in feeds_info_files:
        self._run_feed_keys_info.append(_load_log_message_from_event_file(feeds_info_file))