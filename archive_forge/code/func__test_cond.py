import abc
import itertools
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_v2
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load as load_model
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import save as save_model
from tensorflow.python.util import nest
def _test_cond(use_placeholder, shapes_info, dtype):

    def test_cond(self):
        with self.test_session(graph=ops.Graph()) as sess:
            if 0 in shapes_info.shape[-2:]:
                return
            if test.is_built_with_rocm() and (dtype == dtypes.complex64 or dtype == dtypes.complex128):
                return
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
            operator, mat = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder, ensure_self_adjoint_and_pd=True)
            op_cond = operator.cond()
            s = math_ops.abs(linalg_ops.svd(mat, compute_uv=False))
            mat_cond = math_ops.reduce_max(s, axis=-1) / math_ops.reduce_min(s, axis=-1)
            op_cond_v, mat_cond_v = sess.run([op_cond, mat_cond])
            atol_override = {dtypes.float16: 0.01, dtypes.float32: 0.001, dtypes.float64: 1e-06, dtypes.complex64: 0.001, dtypes.complex128: 1e-06}
            rtol_override = {dtypes.float16: 0.01, dtypes.float32: 0.001, dtypes.float64: 0.0001, dtypes.complex64: 0.001, dtypes.complex128: 1e-06}
            atol = atol_override[dtype]
            rtol = rtol_override[dtype]
            self.assertAllClose(op_cond_v, mat_cond_v, atol=atol, rtol=rtol)
    return test_cond