import abc
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training import slot_creator
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _get_processor(v):
    """The processor of v."""
    if context.executing_eagerly():
        if isinstance(v, tensor.Tensor):
            return _TensorProcessor(v)
        else:
            return _DenseResourceVariableProcessor(v)
    if resource_variable_ops.is_resource_variable(v) and (not v._in_graph_mode):
        return _DenseResourceVariableProcessor(v)
    if v.op.type == 'VarHandleOp':
        return _DenseResourceVariableProcessor(v)
    if isinstance(v, variables.Variable):
        return _RefVariableProcessor(v)
    if isinstance(v, tensor.Tensor):
        return _TensorProcessor(v)
    raise NotImplementedError('Trying to optimize unsupported type ', v)