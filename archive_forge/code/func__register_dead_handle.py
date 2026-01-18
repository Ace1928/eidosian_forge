import collections
import functools
import re
import threading
import warnings
import numpy as np
import wrapt
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import pywrap_tf_session as tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import device
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor
from tensorflow.python.ops import session_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.experimental import mixed_precision_global_state
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _register_dead_handle(self, handle):
    tensors_to_delete = None
    with self._delete_lock:
        self._dead_handles.append(handle)
        if len(self._dead_handles) == BaseSession._DEAD_HANDLES_THRESHOLD:
            tensors_to_delete = self._dead_handles
            self._dead_handles = []
    if tensors_to_delete:
        feeds = {}
        fetches = []
        for deleter_key, tensor_handle in enumerate(tensors_to_delete):
            holder, deleter = session_ops._get_handle_deleter(self.graph, deleter_key, tensor_handle)
            feeds[holder] = tensor_handle
            fetches.append(deleter)
        self.run(fetches, feed_dict=feeds)