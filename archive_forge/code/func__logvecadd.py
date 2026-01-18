import warnings
from Bio import BiopythonDeprecationWarning
def _logvecadd(logvec1, logvec2):
    """Implement a log sum for two vector objects (PRIVATE)."""
    assert len(logvec1) == len(logvec2), "vectors aren't the same length"
    sumvec = np.zeros(len(logvec1))
    for i in range(len(logvec1)):
        sumvec[i] = logaddexp(logvec1[i], logvec2[i])
    return sumvec