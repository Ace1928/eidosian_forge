import cmath
from typing import Tuple
import numpy as np
def CPHASE00(phi: float) -> np.ndarray:
    return np.diag([np.exp(1j * phi), 1.0, 1.0, 1.0])