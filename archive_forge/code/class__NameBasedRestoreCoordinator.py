import abc
import collections
import functools
import glob
import os
import threading
import time
import weakref
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import async_checkpoint_helper
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.checkpoint import graph_view as graph_view_lib
from tensorflow.python.checkpoint import restore as restore_lib
from tensorflow.python.checkpoint import save_util
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import util
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops as io_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import saver as v1_saver_lib
from tensorflow.python.training.saving import saveable_object as saveable_object_lib
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
class _NameBasedRestoreCoordinator:
    """Keeps the status of a name-based checkpoint restore."""

    def __init__(self, save_path, dtype_map=None):
        self.save_path = save_path
        self.dtype_map = dtype_map
        self.unused_attributes = object_identity.ObjectIdentityWeakKeyDictionary()
        self.restore_uid = ops.uid()

    def globally_named_object_attributes(self, trackable):
        """Create globally named SaveableObjects from attributes.

    If an object's attribute has no global name specified (default construction
    for the SaveableObject factory), records the failure in
    `self.unused_attributes` (which can then be used to make status assertions
    fail; see `NameBasedSaverStatus`).

    Args:
      trackable: An object to save.

    Yields:
      SaveableObjects for `trackable`'s attributes.
    """
        for attribute_name, saveable_factory in saveable_object_util.saveable_objects_from_trackable(trackable, tf1_saver=True).items():
            if callable(saveable_factory):
                try:
                    saveable = saveable_factory()
                except TypeError:
                    self.unused_attributes.setdefault(trackable, []).append(attribute_name)
                    continue
            else:
                saveable = saveable_factory
            names_to_saveables = saveable_object_util.op_list_to_dict([saveable], convert_variable_to_tensor=False)
            for name, op in names_to_saveables.items():
                for saveable_object in saveable_object_util.saveable_objects_for_op(op=op, name=name):
                    yield saveable_object

    def eager_restore(self, trackable):
        """Runs restore ops for `trackable`'s attributes."""
        assert context.executing_eagerly()
        for saveable in self.globally_named_object_attributes(trackable):
            restored_tensors = []
            tensor_missing = False
            for spec in saveable.specs:
                if spec.name in self.dtype_map:
                    with ops.device('cpu:0'):
                        restored, = io_ops.restore_v2(prefix=self.save_path, tensor_names=[spec.name], shape_and_slices=[''], dtypes=[self.dtype_map[spec.name]], name='%s_checkpoint_read' % (spec.name,))
                    restored_tensors.append(array_ops.identity(restored))
                else:
                    tensor_missing = True
            if tensor_missing:
                self.unused_attributes.setdefault(trackable, []).append(saveable.name)
            else:
                saveable.restore(restored_tensors=restored_tensors, restored_shapes=None)