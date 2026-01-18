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
class EagerFunc:
    """A wrapper for a function owned by an EagerPyFunc."""

    def __init__(self, func, Tout, is_grad_func):
        """Constructs an EagerFunc.

    Args:
      func: The function to wrap.
      Tout: A list of datatypes for the output; an empty list if the output is
        None.
      is_grad_func: Whether this EagerFunc is the gradient of another
        EagerPyFunc.
    """
        self._func = func
        self._out_dtypes = Tout
        self._is_grad_func = is_grad_func
        self._support_graph_mode_gradient = False

    def set_support_graph_mode_gradient(self):
        """Indicates the object shall support gradient ops.

    This function is internally used by _EagerPyFuncGrad to support
    graph mode gradient of EagerFunc via tf.gradient().
    """
        self._support_graph_mode_gradient = True

    def _convert(self, value, dtype):
        """Converts `value` to a tensor of type `dtype`, with error checking.

    Args:
      value: The tensor to convert.
      dtype: The desired dtype.

    Returns:
      A tensor of type `dtype`, or a zeros tensor if value is None and
      this function is in fact a gradient function.

    Raises:
      RuntimeError: if `value` is a variable.
    """
        if isinstance(value, resource_variable_ops.ResourceVariable):
            raise RuntimeError(f'Attempting to return a variable from an eagerly executed py_func. Only numeric data structures like Tensors or NumPy arrays should be returned; to return the value of a variable, make sure to obtain the Tensor backing it by calling `.read_value()` on the variable in question: {value}')
        if value is None and self._is_grad_func:
            return constant_op.constant(0.0, dtype=dtype)
        return ops.convert_to_tensor(value, dtype=dtype)

    def __call__(self, device, token, args):
        """Calls `self._func` in eager mode, recording the tape if needed."""
        use_tape_cache = self._support_graph_mode_gradient or record.could_possibly_record()
        if use_tape_cache:
            with backprop.GradientTape() as tape:
                for tensor in args:
                    for t in nest.flatten(tensor):
                        if backprop_util.IsTrainable(t):
                            tape.watch(t)
                outputs = self._call(device, args)
            tape_cache[compat.as_bytes(token)] = (tape, args, outputs)
        else:
            outputs = self._call(device, args)
        return outputs

    def _call(self, device, args):
        """Passes `args` to `self._func`, which is executed eagerly."""
        with context.eager_mode():
            ret = self._func(*args)
            device_name = device
            if device_name is None:
                device_name = '/job:localhost/replica:0/task:0/device:CPU:0'
            with ops.device(device):
                if isinstance(ret, (tuple, list)):
                    outputs = [_maybe_copy_to_context_device(self._convert(x, dtype=dtype), device_name) for x, dtype in zip(ret, self._out_dtypes)]
                elif ret is None:
                    outputs = None
                else:
                    outputs = _maybe_copy_to_context_device(self._convert(ret, dtype=self._out_dtypes[0]), device_name)
        return outputs