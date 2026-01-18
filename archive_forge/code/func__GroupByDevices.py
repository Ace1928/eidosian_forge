import collections
import glob
import os.path
import threading
import time
import numpy as np
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import training_util
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
def _GroupByDevices(self, saveables):
    """Group Variable tensor slices per device.

    TODO(touts): Make sure that all the devices found are on different
    job/replica/task/cpu|gpu.  It would be bad if 2 were on the same device.
    It can happen if the devices are unspecified.

    Args:
      saveables: A list of BaseSaverBuilder.SaveableObject objects.

    Returns:
      A list of tuples: (device_name, BaseSaverBuilder.SaveableObject) tuples.
      The list is sorted by ascending device_name.

    Raises:
      ValueError: If the tensors of a saveable are on different devices.
    """
    per_device = collections.defaultdict(lambda: [])
    for saveable in saveables:
        canonical_device = set((pydev.canonical_name(spec.device) for spec in saveable.specs))
        if len(canonical_device) != 1:
            raise ValueError('All tensors of a saveable object must be on the same device: %s' % saveable.name)
        per_device[canonical_device.pop()].append(saveable)
    return sorted(per_device.items(), key=lambda t: t[0])