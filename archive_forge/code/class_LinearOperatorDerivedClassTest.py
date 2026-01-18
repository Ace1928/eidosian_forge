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
class LinearOperatorDerivedClassTest(test.TestCase, metaclass=abc.ABCMeta):
    """Tests for derived classes.

  Subclasses should implement every abstractmethod, and this will enable all
  test methods to work.
  """
    _atol = {dtypes.float16: 0.001, dtypes.float32: 1e-06, dtypes.float64: 1e-12, dtypes.complex64: 1e-06, dtypes.complex128: 1e-12}
    _rtol = {dtypes.float16: 0.001, dtypes.float32: 1e-06, dtypes.float64: 1e-12, dtypes.complex64: 1e-06, dtypes.complex128: 1e-12}

    def assertAC(self, x, y, check_dtype=False):
        """Derived classes can set _atol, _rtol to get different tolerance."""
        dtype = dtypes.as_dtype(x.dtype)
        atol = self._atol[dtype]
        rtol = self._rtol[dtype]
        self.assertAllClose(x, y, atol=atol, rtol=rtol)
        if check_dtype:
            self.assertDTypeEqual(x, y.dtype)

    @staticmethod
    def adjoint_options():
        return [False, True]

    @staticmethod
    def adjoint_arg_options():
        return [False, True]

    @staticmethod
    def dtypes_to_test():
        return [dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128]

    @staticmethod
    def use_placeholder_options():
        return [False, True]

    @staticmethod
    def use_blockwise_arg():
        return False

    @staticmethod
    def operator_shapes_infos():
        """Returns list of OperatorShapesInfo, encapsulating the shape to test."""
        raise NotImplementedError('operator_shapes_infos has not been implemented.')

    @abc.abstractmethod
    def operator_and_matrix(self, shapes_info, dtype, use_placeholder, ensure_self_adjoint_and_pd=False):
        """Build a batch matrix and an Operator that should have similar behavior.

    Every operator acts like a (batch) matrix.  This method returns both
    together, and is used by tests.

    Args:
      shapes_info: `OperatorShapesInfo`, encoding shape information about the
        operator.
      dtype:  Numpy dtype.  Data type of returned array/operator.
      use_placeholder:  Python bool.  If True, initialize the operator with a
        placeholder of undefined shape and correct dtype.
      ensure_self_adjoint_and_pd: If `True`,
        construct this operator to be Hermitian Positive Definite, as well
        as ensuring the hints `is_positive_definite` and `is_self_adjoint`
        are set.
        This is useful for testing methods such as `cholesky`.

    Returns:
      operator:  `LinearOperator` subclass instance.
      mat:  `Tensor` representing operator.
    """
        raise NotImplementedError('Not implemented yet.')

    @abc.abstractmethod
    def make_rhs(self, operator, adjoint, with_batch=True):
        """Make a rhs appropriate for calling operator.solve(rhs).

    Args:
      operator:  A `LinearOperator`
      adjoint:  Python `bool`.  If `True`, we are making a 'rhs' value for the
        adjoint operator.
      with_batch: Python `bool`. If `True`, create `rhs` with the same batch
        shape as operator, and otherwise create a matrix without any batch
        shape.

    Returns:
      A `Tensor`
    """
        raise NotImplementedError('make_rhs is not defined.')

    @abc.abstractmethod
    def make_x(self, operator, adjoint, with_batch=True):
        """Make an 'x' appropriate for calling operator.matmul(x).

    Args:
      operator:  A `LinearOperator`
      adjoint:  Python `bool`.  If `True`, we are making an 'x' value for the
        adjoint operator.
      with_batch: Python `bool`. If `True`, create `x` with the same batch shape
        as operator, and otherwise create a matrix without any batch shape.

    Returns:
      A `Tensor`
    """
        raise NotImplementedError('make_x is not defined.')

    @staticmethod
    def skip_these_tests():
        """List of test names to skip."""
        return []

    @staticmethod
    def optional_tests():
        """List of optional test names to run."""
        return []

    def assertRaisesError(self, msg):
        """assertRaisesRegexp or OpError, depending on context.executing_eagerly."""
        if context.executing_eagerly():
            return self.assertRaisesRegexp(Exception, msg)
        return self.assertRaisesOpError(msg)

    def check_convert_variables_to_tensors(self, operator):
        """Checks that internal Variables are correctly converted to Tensors."""
        self.assertIsInstance(operator, composite_tensor.CompositeTensor)
        tensor_operator = composite_tensor.convert_variables_to_tensors(operator)
        self.assertIs(type(operator), type(tensor_operator))
        self.assertEmpty(tensor_operator.variables)
        self._check_tensors_equal_variables(operator, tensor_operator)

    def _check_tensors_equal_variables(self, obj, tensor_obj):
        """Checks that Variables in `obj` have equivalent Tensors in `tensor_obj."""
        if isinstance(obj, variables.Variable):
            self.assertAllClose(ops.convert_to_tensor(obj), ops.convert_to_tensor(tensor_obj))
        elif isinstance(obj, composite_tensor.CompositeTensor):
            params = getattr(obj, 'parameters', {})
            tensor_params = getattr(tensor_obj, 'parameters', {})
            self.assertAllEqual(params.keys(), tensor_params.keys())
            self._check_tensors_equal_variables(params, tensor_params)
        elif nest.is_mapping(obj):
            for k, v in obj.items():
                self._check_tensors_equal_variables(v, tensor_obj[k])
        elif nest.is_nested(obj):
            for x, y in zip(obj, tensor_obj):
                self._check_tensors_equal_variables(x, y)
        else:
            pass

    def check_tape_safe(self, operator, skip_options=None):
        """Check gradients are not None w.r.t. operator.variables.

    Meant to be called from the derived class.

    This ensures grads are not w.r.t every variable in operator.variables.  If
    more fine-grained testing is needed, a custom test should be written.

    Args:
      operator: LinearOperator.  Exact checks done will depend on hints.
      skip_options: Optional list of CheckTapeSafeSkipOptions.
        Makes this test skip particular checks.
    """
        skip_options = skip_options or []
        if not operator.variables:
            raise AssertionError('`operator.variables` was empty')

        def _assert_not_none(iterable):
            for item in iterable:
                self.assertIsNotNone(item)
        with backprop.GradientTape() as tape:
            grad = tape.gradient(operator.to_dense(), operator.variables)
            _assert_not_none(grad)
        with backprop.GradientTape() as tape:
            var_grad = tape.gradient(operator, operator.variables)
            _assert_not_none(var_grad)
            nest.assert_same_structure(var_grad, grad)
        with backprop.GradientTape() as tape:
            _assert_not_none(tape.gradient(operator.adjoint().to_dense(), operator.variables))
        x = math_ops.cast(array_ops.ones(shape=operator.H.shape_tensor()[:-1]), operator.dtype)
        with backprop.GradientTape() as tape:
            _assert_not_none(tape.gradient(operator.matvec(x), operator.variables))
        if not operator.is_square:
            return
        for option in [CheckTapeSafeSkipOptions.DETERMINANT, CheckTapeSafeSkipOptions.LOG_ABS_DETERMINANT, CheckTapeSafeSkipOptions.DIAG_PART, CheckTapeSafeSkipOptions.TRACE]:
            with backprop.GradientTape() as tape:
                if option not in skip_options:
                    _assert_not_none(tape.gradient(getattr(operator, option)(), operator.variables))
        if operator.is_non_singular is False:
            return
        with backprop.GradientTape() as tape:
            _assert_not_none(tape.gradient(operator.inverse().to_dense(), operator.variables))
        with backprop.GradientTape() as tape:
            _assert_not_none(tape.gradient(operator.solvevec(x), operator.variables))
        if not (operator.is_self_adjoint and operator.is_positive_definite):
            return
        with backprop.GradientTape() as tape:
            _assert_not_none(tape.gradient(operator.cholesky().to_dense(), operator.variables))