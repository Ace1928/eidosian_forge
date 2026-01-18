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
class OpStats(object):
    """A holder for statistics about an operator.

  This class holds information about the resource requirements for an op,
  including the size of its weight parameters on-disk and how many FLOPS it
  requires to execute forward inference.

  If you define a new operation, you can create a function that will return a
  set of information about its usage of the CPU and disk space when serialized.
  The function itself takes a Graph object that's been set up so you can call
  methods like get_tensor_by_name to help calculate the results, and a NodeDef
  argument.

  """
    __slots__ = ['_statistic_type', '_value']

    def __init__(self, statistic_type, value=None) -> None:
        """Sets up the initial placeholders for the statistics."""
        self.statistic_type = statistic_type
        self.value = value

    @property
    def statistic_type(self):
        return self._statistic_type

    @statistic_type.setter
    def statistic_type(self, statistic_type):
        self._statistic_type = statistic_type

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def __iadd__(self, other):
        if other.statistic_type != self.statistic_type:
            raise ValueError("Can't add an OpStat of type %s to one of %s." % (self.statistic_type, other.statistic_type))
        if self.value is None:
            self.value = other.value
        elif other.value is not None:
            self._value += other.value
        return self