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
def _test_solve_base(self, use_placeholder, shapes_info, dtype, adjoint, adjoint_arg, blockwise_arg, with_batch):
    if not with_batch and len(shapes_info.shape) <= 2:
        return
    with self.session(graph=ops.Graph()) as sess:
        sess.graph.seed = random_seed.DEFAULT_GRAPH_SEED
        operator, mat = self.operator_and_matrix(shapes_info, dtype, use_placeholder=use_placeholder)
        rhs = self.make_rhs(operator, adjoint=adjoint, with_batch=with_batch)
        if adjoint_arg:
            op_solve = operator.solve(linalg.adjoint(rhs), adjoint=adjoint, adjoint_arg=adjoint_arg)
        else:
            op_solve = operator.solve(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)
        mat_solve = linear_operator_util.matrix_solve_with_broadcast(mat, rhs, adjoint=adjoint)
        if not use_placeholder:
            self.assertAllEqual(op_solve.shape, mat_solve.shape)
        if blockwise_arg and len(operator.operators) > 1:
            block_dimensions = operator._block_range_dimensions() if adjoint else operator._block_domain_dimensions()
            block_dimensions_fn = operator._block_range_dimension_tensors if adjoint else operator._block_domain_dimension_tensors
            split_rhs = linear_operator_util.split_arg_into_blocks(block_dimensions, block_dimensions_fn, rhs, axis=-2)
            if adjoint_arg:
                split_rhs = [linalg.adjoint(y) for y in split_rhs]
            split_solve = operator.solve(split_rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)
            self.assertEqual(len(split_solve), len(operator.operators))
            split_solve = linear_operator_util.broadcast_matrix_batch_dims(split_solve)
            fused_block_solve = array_ops.concat(split_solve, axis=-2)
            op_solve_v, mat_solve_v, fused_block_solve_v = sess.run([op_solve, mat_solve, fused_block_solve])
            self.assertAC(mat_solve_v, fused_block_solve_v)
        else:
            op_solve_v, mat_solve_v = sess.run([op_solve, mat_solve])
        self.assertAC(op_solve_v, mat_solve_v)