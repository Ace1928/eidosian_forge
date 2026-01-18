import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import kron, eye, dia_array
def _eigenvalue_ordering(self, m):
    """Compute `m` largest eigenvalues in each of the ``N`` directions,
        i.e., up to ``m * N`` total, order them and return `m` largest.
        """
    grid_shape = self.grid_shape
    if m is None:
        indices = np.indices(grid_shape)
        Leig = np.zeros(grid_shape)
    else:
        grid_shape_min = min(grid_shape, tuple(np.ones_like(grid_shape) * m))
        indices = np.indices(grid_shape_min)
        Leig = np.zeros(grid_shape_min)
    for j, n in zip(indices, grid_shape):
        if self.boundary_conditions == 'dirichlet':
            Leig += -4 * np.sin(np.pi * (j + 1) / (2 * (n + 1))) ** 2
        elif self.boundary_conditions == 'neumann':
            Leig += -4 * np.sin(np.pi * j / (2 * n)) ** 2
        else:
            Leig += -4 * np.sin(np.pi * np.floor((j + 1) / 2) / n) ** 2
    Leig_ravel = Leig.ravel()
    ind = np.argsort(Leig_ravel)
    eigenvalues = Leig_ravel[ind]
    if m is not None:
        eigenvalues = eigenvalues[-m:]
        ind = ind[-m:]
    return (eigenvalues, ind)