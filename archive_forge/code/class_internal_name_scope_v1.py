import collections
import copy
import enum
import re
import sys
import threading
import types
from typing import Any, AnyStr, Callable, List, NoReturn, Pattern, Tuple, Type, Union, Optional
from absl import app
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import full_type_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python import pywrap_tfe
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import record
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import registry
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import traceable_stack
from tensorflow.python.framework import versions
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace as profiler_trace
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import lock_util
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_stack
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import kwarg_only
from tensorflow.python.util.tf_export import tf_export
class internal_name_scope_v1(object):
    """Graph-only version of `name_scope_v1`."""

    @property
    def name(self):
        return self._name

    def __init__(self, name, default_name=None, values=None) -> None:
        """Initialize the context manager.

    Args:
      name: The name argument that is passed to the op function.
      default_name: The default name to use if the `name` argument is `None`.
      values: The list of `Tensor` arguments that are passed to the op function.

    Raises:
      TypeError: if `default_name` is passed in but not a string.
    """
        if not (default_name is None or isinstance(default_name, str)):
            raise TypeError('`default_name` type (%s) is not a string type. You likely meant to pass this into the `values` kwarg.' % type(default_name))
        self._name = default_name if name is None else name
        self._default_name = default_name
        self._values = values

    def __enter__(self):
        """Start the scope block.

    Returns:
      The scope name.

    Raises:
      ValueError: if neither `name` nor `default_name` is provided
        but `values` are.
    """
        if self._name is None and self._values is not None:
            raise ValueError('At least one of name (%s) and default_name (%s) must be provided.' % (self._name, self._default_name))
        g = get_default_graph()
        if self._values and (not g.building_function):
            g_from_inputs = _get_graph_from_inputs(self._values)
            if g_from_inputs is not g:
                g = g_from_inputs
                self._g_manager = g.as_default()
                self._g_manager.__enter__()
            else:
                self._g_manager = None
        else:
            self._g_manager = None
        try:
            self._name_scope = g.name_scope(self._name)
            return self._name_scope.__enter__()
        except:
            if self._g_manager is not None:
                self._g_manager.__exit__(*sys.exc_info())
            raise

    def __exit__(self, *exc_info) -> None:
        self._name_scope.__exit__(*exc_info)
        if self._g_manager is not None:
            self._g_manager.__exit__(*exc_info)