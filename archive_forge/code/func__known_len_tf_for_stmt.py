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
def _known_len_tf_for_stmt(iter_, extra_test, body, get_state, set_state, symbol_names, opts):
    """Overload of for_stmt that iterates over TF entities that admit a length."""
    n = py_builtins.len_(iter_)
    ta = tensor_array_ops.TensorArray(iter_.dtype, size=n)
    iter_ = ta.unstack(iter_)
    iterate_index = 0

    def aug_get_state():
        return (iterate_index,) + get_state()

    def aug_set_state(aug_loop_vars):
        nonlocal iterate_index
        iterate_index, *loop_vars = aug_loop_vars
        set_state(loop_vars)

    def aug_body():
        nonlocal iterate_index
        body(iter_.read(iterate_index))
        iterate_index += 1

    def aug_test():
        main_test = iterate_index < n
        if extra_test is not None:
            return tf_cond.cond(main_test, extra_test, lambda: False)
        return main_test
    _add_max_iterations_hint(opts, n)
    _tf_while_stmt(aug_test, aug_body, aug_get_state, aug_set_state, ('<internal iterate>',) + symbol_names, opts)