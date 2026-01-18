import abc
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import control_flow_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.gen_control_flow_ops import *
from tensorflow.python.util import compat
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _init_values_from_proto(self, values_def, import_scope=None):
    """Initializes values and external_values from `ValuesDef` protocol buffer.

    Args:
      values_def: `ValuesDef` protocol buffer.
      import_scope: Optional `string`. Name scope to add.
    """
    assert isinstance(values_def, control_flow_pb2.ValuesDef)
    self._values = set((ops.prepend_name_scope(value, import_scope) for value in values_def.values))
    g = ops.get_default_graph()
    self._external_values = {}
    for k, v in values_def.external_values.items():
        k = ops.prepend_name_scope(k, import_scope)
        self._external_values[k] = g.as_graph_element(ops.prepend_name_scope(v, import_scope))
    op_names = set([op.split(':')[0] for op in self._values - set(self._external_values.keys())])
    for op in op_names:
        g.as_graph_element(op)._set_control_flow_context(self)