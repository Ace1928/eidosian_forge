import numpy as np
import sys
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
import pyaudio
class PhaserEffect:
    """Creates a phaser effect by filtering the sound to create peaks and troughs."""

    def __init__(self, rate: float, depth: float, feedback: float):
        self.rate = rate
        self.depth = depth
        self.feedback = feedback

    def apply_phaser(self, sound: np.ndarray) -> np.ndarray:
        """Applies a phaser effect to the input sound."""
        pass