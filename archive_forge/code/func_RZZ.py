import cmath
from typing import Tuple
import numpy as np
def RZZ(phi: float) -> np.ndarray:
    return np.array([[np.exp(-1j * phi / 2), 0, 0, 0], [0, np.exp(+1j * phi / 2), 0, 0], [0, 0, np.exp(+1j * phi / 2), 0], [0, 0, 0, np.exp(-1j * phi / 2)]])