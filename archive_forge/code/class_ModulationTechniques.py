import numpy as np
import sys
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
import pyaudio
class ModulationTechniques:
    """Applies modulation techniques such as AM, FM, and PM to a sound."""

    def amplitude_modulation(self, carrier: np.ndarray, modulator: np.ndarray, index: float) -> np.ndarray:
        """Performs amplitude modulation on a sound."""
        return carrier * (1 + index * modulator)

    def frequency_modulation(self, carrier: np.ndarray, modulator: np.ndarray, index: float) -> np.ndarray:
        """Performs frequency modulation on a sound."""
        return np.sin(2 * np.pi * carrier * np.cumsum(1 + index * modulator))

    def phase_modulation(self, carrier: np.ndarray, modulator: np.ndarray, index: float) -> np.ndarray:
        """Performs phase modulation on a sound."""
        return np.sin(2 * np.pi * carrier + index * modulator)