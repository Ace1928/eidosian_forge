import cmath
from typing import Tuple
import numpy as np
def PHASE(phi: float) -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, np.exp(1j * phi)]])