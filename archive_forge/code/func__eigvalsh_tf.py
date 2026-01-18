from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _eigvalsh_tf(density_matrix):
    evs = _i('tf').linalg.eigvalsh(density_matrix)
    evs = _i('tf').math.real(evs)
    return evs