import cmath
from typing import Tuple
import numpy as np
def RXX(phi: float) -> np.ndarray:
    return np.array([[np.cos(phi / 2), 0, 0, -1j * np.sin(phi / 2)], [0, np.cos(phi / 2), -1j * np.sin(phi / 2), 0], [0, -1j * np.sin(phi / 2), np.cos(phi / 2), 0], [-1j * np.sin(phi / 2), 0, 0, np.cos(phi / 2)]])