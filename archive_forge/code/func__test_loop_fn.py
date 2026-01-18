import numpy as np
from tensorflow.python.ops import variables
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops
from tensorflow.python.platform import test
from tensorflow.python.util import nest
def _test_loop_fn(self, loop_fn, iters, parallel_iterations=None, fallback_to_while_loop=False, rtol=0.0001, atol=1e-05):
    t1 = pfor_control_flow_ops.pfor(loop_fn, iters=iters, fallback_to_while_loop=fallback_to_while_loop, parallel_iterations=parallel_iterations)
    loop_fn_dtypes = nest.map_structure(lambda x: x.dtype, t1)
    t2 = pfor_control_flow_ops.for_loop(loop_fn, loop_fn_dtypes, iters=iters, parallel_iterations=parallel_iterations)

    def _check_shape(a, b):
        msg = f'Inferred static shapes are different between two loops: {a.shape} vs {b.shape}.'
        if b.shape:
            self.assertEqual(a.shape.as_list()[0], b.shape.as_list()[0], msg)
    nest.map_structure(_check_shape, t1, t2)
    self.run_and_assert_equal(t1, t2, rtol=rtol, atol=atol)