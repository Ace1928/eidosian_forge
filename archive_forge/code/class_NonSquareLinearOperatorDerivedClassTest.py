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
class NonSquareLinearOperatorDerivedClassTest(LinearOperatorDerivedClassTest, metaclass=abc.ABCMeta):
    """Base test class appropriate for generic rectangular operators.

  Square shapes are never tested by this class, so if you want to test your
  operator with a square shape, create two test classes, the other subclassing
  SquareLinearOperatorFullMatrixTest.

  Sub-classes must still define all abstractmethods from
  LinearOperatorDerivedClassTest that are not defined here.
  """

    @staticmethod
    def skip_these_tests():
        """List of test names to skip."""
        return ['cholesky', 'eigvalsh', 'inverse', 'solve', 'solve_with_broadcast', 'det', 'log_abs_det']

    @staticmethod
    def operator_shapes_infos():
        shapes_info = OperatorShapesInfo
        return [shapes_info((2, 1)), shapes_info((1, 2)), shapes_info((1, 3, 2)), shapes_info((3, 3, 4)), shapes_info((2, 1, 2, 4))]

    def make_rhs(self, operator, adjoint, with_batch=True):
        raise NotImplementedError("make_rhs not implemented because we don't test solve")

    def make_x(self, operator, adjoint, with_batch=True):
        r = self._get_num_systems(operator)
        if operator.shape.is_fully_defined():
            batch_shape = operator.batch_shape.as_list()
            if adjoint:
                n = operator.range_dimension.value
            else:
                n = operator.domain_dimension.value
            if with_batch:
                x_shape = batch_shape + [n, r]
            else:
                x_shape = [n, r]
        else:
            batch_shape = operator.batch_shape_tensor()
            if adjoint:
                n = operator.range_dimension_tensor()
            else:
                n = operator.domain_dimension_tensor()
            if with_batch:
                x_shape = array_ops.concat((batch_shape, [n, r]), 0)
            else:
                x_shape = [n, r]
        return random_normal(x_shape, dtype=operator.dtype)

    def _get_num_systems(self, operator):
        """Get some number, either 1 or 2, depending on operator."""
        if operator.tensor_rank is None or operator.tensor_rank % 2:
            return 1
        else:
            return 2