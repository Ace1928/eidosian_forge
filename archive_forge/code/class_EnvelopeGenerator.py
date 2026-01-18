import numpy as np
import sys
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
import pyaudio
class EnvelopeGenerator:
    """Handles the ADSR (Attack, Decay, Sustain, Release) envelope of a sound."""

    def __init__(self, attack: float, decay: float, sustain: float, release: float):
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release

    def apply_envelope(self, sound: np.ndarray) -> np.ndarray:
        """Apply the ADSR envelope to a sound waveform."""
        pass