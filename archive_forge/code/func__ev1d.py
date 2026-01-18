import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import kron, eye, dia_array
def _ev1d(self, j, n):
    """Return 1 eigenvector in 1d with index `j`
        and number of grid points `n` where ``j < n``. 
        """
    if self.boundary_conditions == 'dirichlet':
        i = np.pi * (np.arange(n) + 1) / (n + 1)
        ev = np.sqrt(2.0 / (n + 1.0)) * np.sin(i * (j + 1))
    elif self.boundary_conditions == 'neumann':
        i = np.pi * (np.arange(n) + 0.5) / n
        ev = np.sqrt((1.0 if j == 0 else 2.0) / n) * np.cos(i * j)
    elif j == 0:
        ev = np.sqrt(1.0 / n) * np.ones(n)
    elif j + 1 == n and n % 2 == 0:
        ev = np.sqrt(1.0 / n) * np.tile([1, -1], n // 2)
    else:
        i = 2.0 * np.pi * (np.arange(n) + 0.5) / n
        ev = np.sqrt(2.0 / n) * np.cos(i * np.floor((j + 1) / 2))
    ev[np.abs(ev) < np.finfo(np.float64).eps] = 0.0
    return ev