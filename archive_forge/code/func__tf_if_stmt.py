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
def _tf_if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts):
    """Overload of if_stmt that stages a TF cond."""
    cond = _verify_tf_condition(cond, 'if statement')
    if not nouts:
        prev_get_state, prev_set_state = (get_state, set_state)
        get_state = lambda: (0,) + prev_get_state()
        set_state = lambda v: prev_set_state(v[1:])
        symbol_names += ('<unused dummy>',)
        nouts = 1
    init_vars = get_state()
    new_body_vars_ = [None]
    new_orelse_vars_ = [None]

    def aug_body():
        set_state(init_vars)
        body()
        new_body_vars = get_state()
        new_body_vars = new_body_vars[:nouts]
        new_body_vars_[0] = new_body_vars
        _verify_tf_cond_branch_vars(new_body_vars, symbol_names, 'main')
        if new_orelse_vars_[0] is not None:
            _verify_tf_cond_vars(new_body_vars, new_orelse_vars_[0], symbol_names)
        return new_body_vars

    def aug_orelse():
        set_state(init_vars)
        orelse()
        new_orelse_vars = get_state()
        new_orelse_vars = new_orelse_vars[:nouts]
        new_orelse_vars_[0] = new_orelse_vars
        _verify_tf_cond_branch_vars(new_orelse_vars, symbol_names, 'else')
        if new_body_vars_[0] is not None:
            _verify_tf_cond_vars(new_body_vars_[0], new_orelse_vars, symbol_names)
        return new_orelse_vars
    final_cond_vars = tf_cond.cond(cond, aug_body, aug_orelse, strict=True)
    final_cond_vars = final_cond_vars + init_vars[nouts:]
    set_state(final_cond_vars)