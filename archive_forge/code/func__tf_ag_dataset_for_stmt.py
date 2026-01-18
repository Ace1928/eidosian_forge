from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.experimental.ops import take_while_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def _tf_ag_dataset_for_stmt(ds, extra_test, body, get_state, set_state, symbol_names, opts):
    """Overload of _dataset_for_stmt with early stopping. See for_stmt."""
    init_vars = get_state()
    control_flow.verify_loop_init_vars(init_vars, symbol_names)
    if not init_vars:
        init_vars = (constant_op.constant(0),)
        symbol_names = ('<internal dummy>',)

        def dummy_set_state(unused_dummy):
            pass

        def dummy_get_state():
            return (constant_op.constant(0),)
        get_state, set_state = (dummy_get_state, dummy_set_state)

    def scan_body(scan_state, scan_inputs):
        """Main body of the Dataset.scan."""
        loop_vars, iterate = (scan_state, scan_inputs)
        set_state(loop_vars)

        def main_path():
            body(iterate)
            new_loop_vars = get_state()
            control_flow.verify_tf_loop_vars(init_vars, loop_vars, new_loop_vars, symbol_names, opts, check_shapes=False)
            return new_loop_vars
        if extra_test is not None:
            extra_cond = extra_test()
            new_loop_vars = cond.cond(extra_cond, main_path, lambda: loop_vars)
        else:
            extra_cond = (constant_op.constant(True),)
            new_loop_vars = main_path()
        scan_outputs = (new_loop_vars, extra_cond)
        new_scan_state = new_loop_vars
        return (new_scan_state, scan_outputs)

    def take_while_predicate(unused_loop_vars, extra_cond):
        return extra_cond

    def reduce_body(unused_reduce_state, scan_outputs):
        output_loop_vars, unused_extra_cond = scan_outputs
        new_reduce_state = output_loop_vars
        return new_reduce_state
    ds = _general_purpose_scan(ds, init_vars, scan_body)
    if extra_test is not None:
        ds = ds.apply(take_while_ops.take_while(take_while_predicate))
    final_loop_vars = ds.reduce(init_vars, reduce_body)
    set_state(final_loop_vars)