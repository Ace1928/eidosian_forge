import warnings
from Bio import BiopythonDeprecationWarning
def _copy_and_check(matrix, desired_shape):
    """Copy a matrix and check its dimension. Normalize at the end (PRIVATE)."""
    matrix = np.array(matrix, copy=1)
    if matrix.shape != desired_shape:
        raise ValueError('Incorrect dimension')
    if len(matrix.shape) == 1:
        if np.fabs(sum(matrix) - 1.0) > 0.01:
            raise ValueError('matrix not normalized to 1.0')
    elif len(matrix.shape) == 2:
        for i in range(len(matrix)):
            if np.fabs(sum(matrix[i]) - 1.0) > 0.01:
                raise ValueError('matrix %d not normalized to 1.0' % i)
    else:
        raise ValueError("I don't handle matrices > 2 dimensions")
    return matrix