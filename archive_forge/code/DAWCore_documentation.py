import sys
from PyQt5 import QtWidgets
from DAWModules import SoundModule
from PyQt5.QtCore import Qt
import pyaudio
import numpy as np
from typing import Dict, Type, List, Tuple

        Ensures clean shutdown of the audio stream and PyAudio instance upon closing the application.

        Parameters:
            event (QCloseEvent): The close event.
        