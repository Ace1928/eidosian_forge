import itertools
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_inspect
class RegisterInverse:
    """Decorator to register an Inverse implementation function.

  Usage:

  @linear_operator_algebra.RegisterInverse(lin_op.LinearOperatorIdentity)
  def _inverse_identity(lin_op_a):
    # Return the identity matrix.
  """

    def __init__(self, lin_op_cls_a):
        """Initialize the LinearOperator registrar.

    Args:
      lin_op_cls_a: the class of the LinearOperator to decompose.
    """
        self._key = (lin_op_cls_a,)

    def __call__(self, inverse_fn):
        """Perform the Inverse registration.

    Args:
      inverse_fn: The function to use for the Inverse.

    Returns:
      inverse_fn

    Raises:
      TypeError: if inverse_fn is not a callable.
      ValueError: if a Inverse function has already been registered for
        the given argument classes.
    """
        if not callable(inverse_fn):
            raise TypeError('inverse_fn must be callable, received: {}'.format(inverse_fn))
        if self._key in _INVERSES:
            raise ValueError('Inverse({}) has already been registered to: {}'.format(self._key[0].__name__, _INVERSES[self._key]))
        _INVERSES[self._key] = inverse_fn
        return inverse_fn