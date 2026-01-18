import cmath
from typing import Tuple
import numpy as np
def RX(phi: float) -> np.ndarray:
    return np.array([[np.cos(phi / 2.0), -1j * np.sin(phi / 2.0)], [-1j * np.sin(phi / 2.0), np.cos(phi / 2.0)]])