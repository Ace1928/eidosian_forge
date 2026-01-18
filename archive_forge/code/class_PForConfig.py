import collections
from functools import partial
import string
import sys
import traceback
import numpy as np
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.core.framework import full_type_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import execute
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_switch_case
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
class PForConfig:
    """A configuration object used to communicate with loop body function."""

    def __init__(self):
        self._maybe_iters = None
        self._reduce_map = {}

    def _has_reductions(self):
        """True if some reductions where performed by loop body."""
        return len(self._reduce_map)

    def _set_iters(self, iters):
        """Set number of pfor iterations."""
        if isinstance(iters, tensor_lib.Tensor):
            iters = tensor_util.constant_value(iters)
        self._maybe_iters = iters

    def reduce(self, fn, *args):
        """Performs reduction `fn` on `args` vectorized across pfor iterations.

    Note that `fn` is traced once inside the loop function context. Hence any
    captures or side-effects will happen in that context. Call to the traced
    version of `fn` happens during the construction of the vectorized code.

    Note that this currently may not work inside a control flow construct.
    Args:
      fn: a reduction function. It will be called with arguments that have the
        same structure as *args but with individual values whose rank may be
        higher by 1 since they represent loop invariant vectorized versions of
        the corresponding Tensors in *args.
      *args: unvectorized Tensors.

    Returns:
      The result of running `fn` on the vectorized versions of `*args`. These
      outputs will be available as loop invariant values to all the iterations.
    """
        assert not context.executing_eagerly()
        tensor_specs = []
        for arg in args:
            if not isinstance(arg, tensor_lib.Tensor):
                raise ValueError(f'Got a non-Tensor argument {arg} in reduce.')
            batched_shape = tensor_shape.TensorShape([self._maybe_iters]).concatenate(arg.shape)
            tensor_specs.append(tensor_lib.TensorSpec(shape=batched_shape, dtype=arg.dtype))
        concrete_function = def_function.function(fn).get_concrete_function(*tensor_specs)
        pl_outputs = []
        with ops.control_dependencies(args):
            for output in concrete_function.outputs:
                if not isinstance(output, tensor_lib.Tensor):
                    raise ValueError(f'Got a non-Tensor output {output} while running reduce.')
                if output.shape.is_fully_defined():
                    dummy = array_ops.zeros(output.shape.as_list(), dtype=output.dtype)
                    pl_outputs.append(array_ops.placeholder_with_default(dummy, shape=output.shape))
                else:
                    pl_outputs.append(array_ops.placeholder(output.dtype, shape=output.shape))
            reduction_op = array_ops.identity_n(pl_outputs)[0].op
        self._reduce_map[reduction_op] = (concrete_function, args)
        if len(reduction_op.outputs) == 1:
            return reduction_op.outputs[0]
        else:
            return tuple(reduction_op.outputs)

    def reduce_concat(self, x):
        """Performs a concat reduction on `x` across pfor iterations.

    Note that this currently may not work inside a control flow construct.
    Args:
      x: an unvectorized Tensor.

    Returns:
      A Tensor that has rank one higher than `x`. The value is the vectorized
      version of `x`, i.e. stacking the value of `x` across different pfor
      iterations.
    """
        return self.reduce(lambda y: y, x)

    def reduce_mean(self, x):
        """Performs a mean reduction on `x` across pfor iterations.

    Note that this currently may not work inside a control flow construct.
    Args:
      x: an unvectorized Tensor.

    Returns:
      A Tensor that has same rank as `x`. The value is the mean of the values
      of `x` across the pfor iterations.
    """
        return self.reduce(lambda y: math_ops.reduce_mean(y, axis=0), x)

    def reduce_sum(self, x):
        """Performs a sum reduction on `x` across pfor iterations.

    Note that this currently may not work inside a control flow construct.
    Args:
      x: an unvectorized Tensor.

    Returns:
      A Tensor that has same rank as `x`. The value is the sum of the values
      of `x` across the pfor iterations.
    """
        return self.reduce(lambda y: math_ops.reduce_sum(y, axis=0), x)

    def _lookup_reduction(self, t):
        """Lookups Tensor `t` in the reduction maps."""
        assert isinstance(t, tensor_lib.Tensor), t
        return self._reduce_map.get(t.op)