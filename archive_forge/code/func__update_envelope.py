from dataclasses import dataclass
from fractions import Fraction
from numbers import Complex
from typing import (
import numpy as np
def _update_envelope(iqs: np.ndarray, rate: float, scale: Optional[float], phase: Optional[float], detuning: Optional[float]) -> np.ndarray:
    """Update a pulse envelope by optional shape parameters.

    The optional parameters are: 'scale', 'phase', 'detuning'.

    :param iqs: The basic pulse envelope.
    :param rate: The sample rate (in Hz).
    :return: The updated pulse envelope.
    """

    def default(obj: Optional[float], val: float) -> float:
        return obj if obj is not None else val
    scale = default(scale, 1.0)
    phase = default(phase, 0.0)
    detuning = default(detuning, 0.0)
    iqs *= scale * np.exp(1j * phase) * np.exp(1j * 2 * np.pi * detuning * np.arange(len(iqs)) / rate)
    return iqs