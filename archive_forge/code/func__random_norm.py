import warnings
from Bio import BiopythonDeprecationWarning
def _random_norm(shape):
    """Normalize a random matrix (PRIVATE)."""
    matrix = np.random.random(shape)
    return _normalize(matrix)