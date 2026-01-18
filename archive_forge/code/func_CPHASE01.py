import cmath
from typing import Tuple
import numpy as np
def CPHASE01(phi: float) -> np.ndarray:
    return np.diag([1.0, np.exp(1j * phi), 1.0, 1.0])