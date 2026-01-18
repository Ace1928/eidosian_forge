import cmath
from typing import Tuple
import numpy as np
def RY(phi: float) -> np.ndarray:
    return np.array([[np.cos(phi / 2.0), -np.sin(phi / 2.0)], [np.sin(phi / 2.0), np.cos(phi / 2.0)]])