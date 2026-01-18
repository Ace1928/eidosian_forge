import abc
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_full_matrix
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_lower_triangular
class _Adder(metaclass=abc.ABCMeta):
    """Abstract base class to add two operators.

  Each `Adder` acts independently, adding everything it can, paying no attention
  as to whether another `Adder` could have done the addition more efficiently.
  """

    @property
    def name(self):
        return self.__class__.__name__

    @abc.abstractmethod
    def can_add(self, op1, op2):
        """Returns `True` if this `Adder` can add `op1` and `op2`.  Else `False`."""
        pass

    @abc.abstractmethod
    def _add(self, op1, op2, operator_name, hints):
        pass

    def add(self, op1, op2, operator_name, hints=None):
        """Return new `LinearOperator` acting like `op1 + op2`.

    Args:
      op1:  `LinearOperator`
      op2:  `LinearOperator`, with `shape` and `dtype` such that adding to
        `op1` is allowed.
      operator_name:  `String` name to give to returned `LinearOperator`
      hints:  `_Hints` object.  Returned `LinearOperator` will be created with
        these hints.

    Returns:
      `LinearOperator`
    """
        updated_hints = _infer_hints_allowing_override(op1, op2, hints)
        if operator_name is None:
            operator_name = 'Add/' + op1.name + '__' + op2.name + '/'
        scope_name = self.name
        if scope_name.startswith('_'):
            scope_name = scope_name[1:]
        with ops.name_scope(scope_name):
            return self._add(op1, op2, operator_name, updated_hints)