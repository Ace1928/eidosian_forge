import numpy as np
import sys
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
import pyaudio
class HarmonicGenerator:
    """Generates and manipulates overtones above the fundamental frequency."""

    def __init__(self, fundamental_frequency: float):
        self.fundamental_frequency = fundamental_frequency
        self.overtones = {}

    def add_overtones(self, number_of_overtones: int) -> None:
        """Generates a specified number of overtones based on the fundamental frequency."""
        pass

    def modify_overtones(self, overtone_id: int, amplitude: float) -> None:
        """Modifies the amplitude of a specific overtone."""
        pass