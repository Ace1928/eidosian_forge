import cmath
from typing import Tuple
import numpy as np
def PHASEDFSIM(theta: float, zeta: float, chi: float, gamma: float, phi: float) -> np.ndarray:
    return np.array([[1, 0, 0, 0], [0, np.exp(-1j * (gamma + zeta)) * np.cos(theta / 2), 1j * np.exp(-1j * (gamma - chi)) * np.sin(theta / 2), 0], [0, 1j * np.exp(-1j * (gamma + chi)) * np.sin(theta / 2), np.exp(-1j * (gamma - zeta)) * np.cos(theta / 2), 0], [0, 0, 0, np.exp(1j * phi - 2j * gamma)]])