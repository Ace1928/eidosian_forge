import numpy as np
import sys
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
import pyaudio
class SampleRateAdjustment:
    """Adjusts the sample rate of a digital sound signal."""

    def resample(self, sound: np.ndarray, new_rate: int) -> np.ndarray:
        """Resamples the sound to a new sample rate."""
        pass