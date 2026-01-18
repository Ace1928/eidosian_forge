import functools
import sys
import traceback
import numpy as np
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.autograph.operators import variables
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.autograph.utils import misc
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.autograph.utils import type_registry
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.types import distribute
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
def _tf_range_for_stmt(iter_, extra_test, body, get_state, set_state, symbol_names, opts):
    """Overload of for_stmt that iterates over a TF range (and elides it)."""
    start, limit, delta = iter_.op.inputs
    iterate = start

    def _value_or(name, var, default):
        if name == opts['iterate_names'] and isinstance(var, variables.Undefined):
            return default
        return var

    def aug_get_state():
        state_vars = get_state()
        state_vars = tuple((_value_or(name, var, iterate) for name, var in zip(symbol_names, state_vars)))
        return (iterate,) + state_vars

    def aug_set_state(aug_loop_vars):
        nonlocal iterate
        iterate, *loop_vars = aug_loop_vars
        set_state(loop_vars)

    def aug_body():
        nonlocal iterate
        body(iterate)
        iterate += delta

    def aug_test():
        const_delta = tensor_util.constant_value(delta)
        if const_delta is not None:
            if const_delta >= 0:
                main_test = iterate < limit
            else:
                main_test = iterate > limit
        else:
            main_test = math_ops.logical_or(math_ops.logical_and(delta >= 0, iterate < limit), math_ops.logical_and(delta < 0, iterate > limit))
        if extra_test is not None:
            main_test = tf_cond.cond(main_test, extra_test, lambda: False)
        return main_test
    _add_max_iterations_hint(opts, math_ops.cast(misc.get_range_len(start, limit, delta), dtypes.int32))
    _tf_while_stmt(aug_test, aug_body, aug_get_state, aug_set_state, ('<internal iterate>',) + symbol_names, opts)