import collections
import contextlib
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_state
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util import variable_utils
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
def _MaybeCompile(scope, op, func, grad_fn):
    """Compile the calculation in grad_fn if op was marked as compiled."""
    scope = scope.rstrip('/').replace('/', '_')
    if func is not None:
        xla_compile = func.cached_definition.attr['_XlaCompile'].b
        xla_separate_compiled_gradients = func.cached_definition.attr['_XlaSeparateCompiledGradients'].b
        xla_scope = func.cached_definition.attr['_XlaScope'].s.decode()
    else:
        try:
            xla_compile = op.get_attr('_XlaCompile')
            xla_separate_compiled_gradients = op.get_attr('_XlaSeparateCompiledGradients')
            xla_scope = op.get_attr('_XlaScope').decode()
        except ValueError:
            xla_compile = False
    if not xla_compile:
        return grad_fn()
    if xla_separate_compiled_gradients:
        xla_grad_scope = '%s_grad_%s' % (xla_scope, scope)
    else:
        xla_grad_scope = xla_scope
    attrs = {'_XlaCompile': attr_value_pb2.AttrValue(b=xla_compile), '_XlaScope': attr_value_pb2.AttrValue(s=xla_grad_scope.encode())}
    with ops.get_default_graph()._attr_scope(attrs):
        return grad_fn()