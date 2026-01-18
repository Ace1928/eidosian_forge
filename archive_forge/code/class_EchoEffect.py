import numpy as np
import sys
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
import pyaudio
class EchoEffect:
    """Generates echo effects by delaying and replaying the sound."""

    def __init__(self, delay_time: float, feedback: float):
        self.delay_time = delay_time
        self.feedback = feedback

    def apply_echo(self, sound: np.ndarray) -> np.ndarray:
        """Applies an echo effect using the specified delay time and feedback."""
        pass