import numpy as np
from scipy import linalg as LA
from cvxpy.atoms.lambda_max import lambda_max
from cvxpy.atoms.sum_largest import sum_largest
class lambda_sum_largest(lambda_max):
    """Sum of the largest k eigenvalues.
    """
    _allow_complex = True

    def __init__(self, X, k) -> None:
        self.k = k
        super(lambda_sum_largest, self).__init__(X)

    def validate_arguments(self) -> None:
        """Verify that the argument A is square.
        """
        X = self.args[0]
        if not X.ndim == 2 or X.shape[0] != X.shape[1]:
            raise ValueError('First argument must be a square matrix.')
        elif int(self.k) != self.k or self.k <= 0:
            raise ValueError('Second argument must be a positive integer.')

    def numeric(self, values):
        """Returns the largest eigenvalue of A.

        Requires that A be symmetric.
        """
        eigs = LA.eigvalsh(values[0])
        return sum_largest(eigs, self.k).value

    def get_data(self):
        """Returns the parameter k.
        """
        return [self.k]

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        raise NotImplementedError()

    @property
    def value(self):
        if not np.allclose(self.args[0].value, self.args[0].value.T.conj()):
            raise ValueError('Input matrix was not Hermitian/symmetric.')
        if any([p.value is None for p in self.parameters()]):
            return None
        return self._value_impl()