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
def _default_getter(name, shape, dtype, initializer=None, partition_info=None, **kwargs):
    """A pared-down version of get_variable which does not reuse variables."""
    dtype = dtypes.as_dtype(dtype)
    shape_object = tensor_shape.as_shape(shape)
    with ops.init_scope():
        if initializer is None:
            initializer, initializing_from_value = variable_scope._get_default_variable_store()._get_default_initializer(name=name, shape=shape_object, dtype=dtype)
        else:
            initializing_from_value = not callable(initializer)
        variable_dtype = dtype.base_dtype
        if initializing_from_value:
            if shape is not None:
                raise ValueError('If initializer is a constant, do not specify shape.')
            initial_value = initializer
        else:
            if isinstance(initializer, type(init_ops.Initializer)):
                initializer = initializer(dtype=dtype)
            shape_list = None if shape is None else shape_object.as_list()
            if 'partition_info' in tf_inspect.getargspec(initializer).args:
                initial_value = functools.partial(initializer, shape_list, dtype=dtype, partition_info=partition_info)
            else:
                initial_value = functools.partial(initializer, shape_list, dtype=dtype)
        return variable_v1.VariableV1(initial_value=initial_value, name=name, dtype=variable_dtype, use_resource=True, **kwargs)