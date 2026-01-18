import numpy as np
import sys
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
import pyaudio
class SoundModule:
    """
    Abstract base class for all sound modules in the DAW.
    This class defines the interface and common functionality across all sound modules.
    """

    def __init__(self):
        pass

    def process_sound(self, sound: np.ndarray) -> np.ndarray:
        """
        Process the sound data. Must be implemented by each module to modify the audio signal.

        Parameters:
            sound (np.ndarray): The input sound data as a NumPy array.

        Returns:
            np.ndarray: The processed sound data.
        """
        raise NotImplementedError('Each module must implement the process_sound method.')

    def set_parameter(self, parameter: str, value: float):
        """
        Set parameters for the sound module. Should be implemented by modules that have parameters.

        Parameters:
            parameter (str): The name of the parameter to set.
            value (float): The value to set the parameter to.
        """
        raise NotImplementedError('This method should be overridden by modules that have parameters.')