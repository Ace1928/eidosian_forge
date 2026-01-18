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
def _select_signal_processing_mode(self, signals, simulation_params):
    if simulation_params.signal_processing_mode in ['advanced', 'complex']:
        return self.advanced_signal_processing(signals, simulation_params)
    else:
        return self.basic_signal_processing(signals, simulation_params)