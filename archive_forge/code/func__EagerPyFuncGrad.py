import functools
import threading
import traceback  # pylint: disable=unused-import
import weakref
import numpy as np
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.eager import backprop
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.lib.core import _pywrap_py_func
from tensorflow.python.ops import autograph_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gen_script_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
@ops.RegisterGradient('EagerPyFunc')
def _EagerPyFuncGrad(op, *dy):
    """Computes the gradient of an EagerPyFunc."""
    token = op.get_attr('token')

    def eagerly_executed_grad(*dy):
        tape, eager_inputs, eager_outputs = tape_cache.pop(compat.as_bytes(token))
        return tape.gradient(eager_outputs, eager_inputs, output_gradients=dy)
    with ops.control_dependencies(op.outputs):
        gradient_op = _internal_py_func(func=eagerly_executed_grad, inp=dy, Tout=[tensor.dtype for tensor in op.inputs], use_eager_py_func=True, is_grad_func=True)
    if not context.executing_eagerly():
        func = _py_funcs.get(token.decode())
        assert isinstance(func, EagerFunc), f'EagerPyFuncGrad called on a non-EagerFunc object: {func}.'
        func.set_support_graph_mode_gradient()
    return gradient_op