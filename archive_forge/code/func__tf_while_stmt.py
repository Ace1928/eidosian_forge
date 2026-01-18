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
def _tf_while_stmt(test, body, get_state, set_state, symbol_names, opts):
    """Overload of while_stmt that stages a TF while_stmt."""
    init_vars = get_state()
    orig_init_vars = init_vars
    nulls = tuple((_is_none_or_undef(v) for v in init_vars))
    if any(nulls):
        shape_invars_by_init_vals = {id(v): i for v, i in opts.get('shape_invariants', ())}
        shape_invariants = tuple((shape_invars_by_init_vals.get(id(v), None) for v in orig_init_vars))
        require_one_iteration, init_vars, extra_shape_invariants = _try_handling_undefineds(body, get_state, set_state, init_vars, nulls, shape_invariants, symbol_names)
    else:
        require_one_iteration = False
    if require_one_iteration:
        merged_shape_invariants = dict(shape_invars_by_init_vals)
        for v, nv, ni in zip(orig_init_vars, init_vars, extra_shape_invariants):
            merged_invariant = merged_shape_invariants.get(id(v), ni)
            if merged_invariant is not None:
                merged_shape_invariants[id(nv)] = merged_invariant
        merged_shape_invariants = tuple(((nv, merged_shape_invariants[id(nv)]) for nv in init_vars if id(nv) in merged_shape_invariants))
        if merged_shape_invariants:
            opts = dict(**opts)
            opts['shape_invariants'] = merged_shape_invariants

    def aug_test(*loop_vars):
        if require_one_iteration:
            loop_vars = loop_vars[1:]
        set_state(loop_vars)
        return _verify_tf_condition(test(), 'while loop')

    def aug_body(*loop_vars):
        if require_one_iteration:
            loop_vars = loop_vars[1:]
        set_state(loop_vars)
        body()
        new_loop_vars = get_state()
        verify_tf_loop_vars(init_vars, loop_vars, new_loop_vars, symbol_names, opts)
        if require_one_iteration:
            new_loop_vars = (True,) + new_loop_vars
        return new_loop_vars
    if 'shape_invariants' in opts:
        opts['shape_invariants'] = _shape_invariants_mapping_to_positional_list(opts['shape_invariants'], init_vars)
    while_loop_opts = dict(opts)
    while_loop_opts.pop('iterate_names', None)
    while_loop_opts['return_same_structure'] = True
    if require_one_iteration:
        aug_init_vars = (False,) + init_vars
        if 'shape_invariants' in while_loop_opts:
            while_loop_opts['shape_invariants'] = (None,) + while_loop_opts['shape_invariants']
    else:
        aug_init_vars = init_vars
    final_loop_vars = while_loop.while_loop(aug_test, aug_body, aug_init_vars, **while_loop_opts)
    if require_one_iteration:
        with ops.control_dependencies([control_flow_assert.Assert(final_loop_vars[0], [_runtime_zero_iterations_errmsg(symbol_names, nulls, orig_init_vars)])]):
            final_loop_vars = nest.map_structure(lambda v: array_ops.identity(v) if tensor_util.is_tf_type(v) else v, final_loop_vars[1:])
    set_state(final_loop_vars)