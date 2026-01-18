from __future__ import annotations
import re
from fractions import Fraction
from typing import Any
import numpy as np
from qiskit import pulse, circuit
from qiskit.pulse import instructions, library
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import drawings, types, device_info
def _find_consecutive_index(data_array: np.ndarray, resolution: float) -> np.ndarray:
    """A helper function to return non-consecutive index from the given list.

    This drastically reduces memory footprint to represent a drawing,
    especially for samples of very long flat-topped Gaussian pulses.
    Tiny value fluctuation smaller than `resolution` threshold is removed.

    Args:
        data_array: The array of numbers.
        resolution: Minimum resolution of sample values.

    Returns:
        The compressed data array.
    """
    try:
        vector = np.asarray(data_array, dtype=float)
        diff = np.diff(vector)
        diff[np.where(np.abs(diff) < resolution)] = 0
        consecutive_l = np.insert(diff.astype(bool), 0, True)
        consecutive_r = np.append(diff.astype(bool), True)
        return consecutive_l | consecutive_r
    except ValueError:
        return np.ones_like(data_array).astype(bool)