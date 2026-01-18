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
def _tf_iterator_for_stmt(iter_, extra_test, body, get_state, set_state, symbol_names, opts):
    """Overload of for_stmt that iterates over TF Iterators. See for_loop."""
    symbol_names = ('<internal has_next>',) + symbol_names
    has_next = True

    def aug_get_state():
        return (has_next,) + get_state()

    def aug_set_state(aug_loop_vars):
        nonlocal has_next
        has_next, *loop_vars = aug_loop_vars
        set_state(loop_vars)
    init_vars = aug_get_state()
    verify_loop_init_vars(init_vars, symbol_names)

    def aug_body():
        """Main body passed to _tf_while_stmt."""
        nonlocal has_next
        opt_iterate = iter_.get_next_as_optional()
        has_next = opt_iterate.has_value()
        loop_vars = aug_get_state()

        def main_path():
            body(opt_iterate.get_value())
            new_loop_vars = aug_get_state()
            verify_tf_loop_vars(init_vars, loop_vars, new_loop_vars, symbol_names, opts)
            return new_loop_vars

        def noop_path():
            return loop_vars
        aug_set_state(tf_cond.cond(has_next, main_path, noop_path))

    def aug_test():
        main_test = has_next
        if extra_test is not None:
            return tf_cond.cond(main_test, extra_test, lambda: False)
        return main_test
    _tf_while_stmt(aug_test, aug_body, aug_get_state, aug_set_state, symbol_names, opts)