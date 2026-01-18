from functools import reduce
import warnings
from Bio import BiopythonDeprecationWarning
def _calc_f_sharp(N, nclasses, features):
    """Calculate a matrix of f sharp values (PRIVATE)."""
    f_sharp = np.zeros((N, nclasses))
    for feature in features:
        for (i, j), f in feature.items():
            f_sharp[i][j] += f
    return f_sharp