import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from collections import deque
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox
from matplotlib.animation import FuncAnimation
import logging
import datetime
import sys
import cProfile
def _validate_signals(self, signals):
    if not isinstance(signals, np.ndarray):
        raise TypeError('Signals must be a NumPy array')
    if signals.ndim != 2:
        raise ValueError('Signals array must be 2-dimensional')