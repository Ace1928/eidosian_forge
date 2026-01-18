import itertools
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_inspect
class RegisterMatmul:
    """Decorator to register a Matmul implementation function.

  Usage:

  @linear_operator_algebra.RegisterMatmul(
    lin_op.LinearOperatorIdentity,
    lin_op.LinearOperatorIdentity)
  def _matmul_identity(a, b):
    # Return the identity matrix.
  """

    def __init__(self, lin_op_cls_a, lin_op_cls_b):
        """Initialize the LinearOperator registrar.

    Args:
      lin_op_cls_a: the class of the LinearOperator to multiply.
      lin_op_cls_b: the class of the second LinearOperator to multiply.
    """
        self._key = (lin_op_cls_a, lin_op_cls_b)

    def __call__(self, matmul_fn):
        """Perform the Matmul registration.

    Args:
      matmul_fn: The function to use for the Matmul.

    Returns:
      matmul_fn

    Raises:
      TypeError: if matmul_fn is not a callable.
      ValueError: if a Matmul function has already been registered for
        the given argument classes.
    """
        if not callable(matmul_fn):
            raise TypeError('matmul_fn must be callable, received: {}'.format(matmul_fn))
        if self._key in _MATMUL:
            raise ValueError('Matmul({}, {}) has already been registered.'.format(self._key[0].__name__, self._key[1].__name__))
        _MATMUL[self._key] = matmul_fn
        return matmul_fn