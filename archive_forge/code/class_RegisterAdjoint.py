import itertools
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_inspect
class RegisterAdjoint:
    """Decorator to register an Adjoint implementation function.

  Usage:

  @linear_operator_algebra.RegisterAdjoint(lin_op.LinearOperatorIdentity)
  def _adjoint_identity(lin_op_a):
    # Return the identity matrix.
  """

    def __init__(self, lin_op_cls_a):
        """Initialize the LinearOperator registrar.

    Args:
      lin_op_cls_a: the class of the LinearOperator to decompose.
    """
        self._key = (lin_op_cls_a,)

    def __call__(self, adjoint_fn):
        """Perform the Adjoint registration.

    Args:
      adjoint_fn: The function to use for the Adjoint.

    Returns:
      adjoint_fn

    Raises:
      TypeError: if adjoint_fn is not a callable.
      ValueError: if a Adjoint function has already been registered for
        the given argument classes.
    """
        if not callable(adjoint_fn):
            raise TypeError('adjoint_fn must be callable, received: {}'.format(adjoint_fn))
        if self._key in _ADJOINTS:
            raise ValueError('Adjoint({}) has already been registered to: {}'.format(self._key[0].__name__, _ADJOINTS[self._key]))
        _ADJOINTS[self._key] = adjoint_fn
        return adjoint_fn