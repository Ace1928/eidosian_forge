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
def _test_operator_solve_with_same_type(use_placeholder, shapes_info, dtype):
    """op_a.solve(op_b), in the case where the same type is returned."""

    def test_operator_solve_with_same_type(self):
        with self.session(graph=ops.Graph()) as sess:
            sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
            operator_a, mat_a = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
            operator_b, mat_b = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
            mat_solve = linear_operator_util.matrix_solve_with_broadcast(mat_a, mat_b)
            op_solve = operator_a.solve(operator_b)
            mat_solve_v, op_solve_v = sess.run([mat_solve, op_solve.to_dense()])
            self.assertIsInstance(op_solve, operator_a.__class__)
            self.assertAC(mat_solve_v, op_solve_v)
    return test_operator_solve_with_same_type