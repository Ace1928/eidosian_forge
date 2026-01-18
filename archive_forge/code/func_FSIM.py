import cmath
from typing import Tuple
import numpy as np
def FSIM(theta: float, phi: float) -> np.ndarray:
    return np.array([[1, 0, 0, 0], [0, np.cos(theta / 2), 1j * np.sin(theta / 2), 0], [0, 1j * np.sin(theta / 2), np.cos(theta / 2), 0], [0, 0, 0, np.exp(1j * phi)]])