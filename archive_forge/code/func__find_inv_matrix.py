import dataclasses
import itertools
from typing import Any, cast, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from cirq import circuits, ops, protocols
def _find_inv_matrix(mat: np.ndarray, mat_sequence: np.ndarray) -> int:
    mat_prod = np.einsum('ij,...jk->...ik', mat, mat_sequence)
    diag_sums = list(np.absolute(np.einsum('...ii->...', mat_prod)))
    idx = diag_sums.index(max(diag_sums))
    return idx