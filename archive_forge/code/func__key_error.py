import math
import warnings
from Bio import SeqUtils, Seq
from Bio import BiopythonWarning
def _key_error(neighbors, strict):
    """Throw an error or a warning if there is no data for the neighbors (PRIVATE)."""
    if strict:
        raise ValueError(f'no thermodynamic data for neighbors {neighbors!r} available')
    else:
        warnings.warn('no themodynamic data for neighbors %r available. Calculation will be wrong' % neighbors, BiopythonWarning)