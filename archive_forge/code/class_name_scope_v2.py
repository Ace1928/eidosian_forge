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
@tf_export('name_scope', v1=[])
class name_scope_v2(object):
    """A context manager for use when defining a Python op.

  This context manager pushes a name scope, which will make the name of all
  operations added within it have a prefix.

  For example, to define a new Python op called `my_op`:

  ```python
  def my_op(a, b, c, name=None):
    with tf.name_scope("MyOp") as scope:
      a = tf.convert_to_tensor(a, name="a")
      b = tf.convert_to_tensor(b, name="b")
      c = tf.convert_to_tensor(c, name="c")
      # Define some computation that uses `a`, `b`, and `c`.
      return foo_op(..., name=scope)
  ```

  When executed, the Tensors `a`, `b`, `c`, will have names `MyOp/a`, `MyOp/b`,
  and `MyOp/c`.

  Inside a `tf.function`, if the scope name already exists, the name will be
  made unique by appending `_n`. For example, calling `my_op` the second time
  will generate `MyOp_1/a`, etc.
  """
    __slots__ = ['_name', '_exit_fns']

    def __init__(self, name):
        """Initialize the context manager.

    Args:
      name: The prefix to use on all names created within the name scope.

    Raises:
      ValueError: If name is not a string.
    """
        if not isinstance(name, str):
            raise ValueError('name for name_scope must be a string.')
        self._name = name
        self._exit_fns = []

    @property
    def name(self):
        return self._name

    def __enter__(self):
        """Start the scope block.

    Returns:
      The scope name.
    """
        ctx = context.context()
        if ctx.executing_eagerly():
            old_name = ctx.scope_name
            name = self._name
            if not name:
                scope_name = ''
            elif name[-1] == '/':
                scope_name = name
            elif old_name:
                scope_name = old_name + name + '/'
            else:
                scope_name = name + '/'
            ctx.scope_name = scope_name

            def _restore_name_scope(*_):
                ctx.scope_name = old_name
            self._exit_fns.append(_restore_name_scope)
        else:
            scope = get_default_graph().name_scope(self._name)
            scope_name = scope.__enter__()
            self._exit_fns.append(scope.__exit__)
        return scope_name

    def __exit__(self, type_arg, value_arg, traceback_arg):
        self._exit_fns.pop()(type_arg, value_arg, traceback_arg)
        return False

    def __getstate__(self):
        return (self._name, self._exit_fns)

    def __setstate__(self, state):
        self._name = state[0]
        self._exit_fns = state[1]