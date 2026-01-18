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
def _create_op_helper(self, op, compute_device=True):
    """Common logic for creating an op in this graph."""
    for key, value in self._attr_scope_map.items():
        try:
            op.get_attr(key)
        except ValueError:
            if callable(value):
                value = value(op.node_def)
                if not isinstance(value, (type(None), attr_value_pb2.AttrValue)):
                    raise TypeError("Callable for scope map key '%s' must return either None or an AttrValue protocol buffer; but it returned: %s" % (key, value))
            if value:
                op._set_attr(key, value)
    try:
        kernel_label = self._op_to_kernel_label_map[op.type]
        op._set_attr('_kernel', attr_value_pb2.AttrValue(s=compat.as_bytes(kernel_label)))
    except KeyError:
        pass
    op._gradient_function = self._gradient_function_map.get(op.type)
    try:
        mapped_op_type = self._gradient_override_map[op.type]
        op._set_attr('_gradient_op_type', attr_value_pb2.AttrValue(s=compat.as_bytes(mapped_op_type)))
    except KeyError:
        pass
    self._record_op_seen_by_control_dependencies(op)
    if compute_device:
        self._apply_device_functions(op)
    op._colocation_code_locations = self._snapshot_colocation_stack_metadata()
    if self._colocation_stack:
        all_colocation_groups = []
        is_device_set = False
        for colocation_op in self._colocation_stack.peek_objs():
            try:
                all_colocation_groups.extend(colocation_op.colocation_groups())
            except AttributeError:
                pass
            if colocation_op.device and (not is_device_set):
                op._set_device(colocation_op.device)
                is_device_set = True
        all_colocation_groups = sorted(set(all_colocation_groups))
        op._set_attr('_class', attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(s=all_colocation_groups)))
    if self._container and op._is_stateful:
        try:
            container_attr = op.get_attr('container')
        except ValueError:
            pass
        else:
            if not container_attr:
                op._set_attr('container', attr_value_pb2.AttrValue(s=compat.as_bytes(self._container)))