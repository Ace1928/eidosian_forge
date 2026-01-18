import numpy as np
import sys
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
import pyaudio
class FlangerEffect:
    """Creates a flanging effect by mixing the sound with a delayed version of itself."""

    def __init__(self, delay: float, depth: float, rate: float, feedback: float):
        self.delay = delay
        self.depth = depth
        self.rate = rate
        self.feedback = feedback

    def apply_flanger(self, sound: np.ndarray) -> np.ndarray:
        """Applies a flanger effect to the sound."""
        pass