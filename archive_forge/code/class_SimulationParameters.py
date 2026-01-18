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
class SimulationParameters:

    def __init__(self):
        self.signal_processing_mode = 'complex'
        self.signal_amplification = 1.0
        self.pattern_detection_sensitivity = 1.0
        self.filter_params = {'filter_type': 'low_pass', 'cutoff_frequency': 100}
        self.wavelet_params = {'wavelet_type': 'db1', 'level': 2}
        self.pattern_params = {'max_score': 50}
        self.mapping_params = {'steepness': 1.0, 'skew_factor': 1.0}
        self.output_params = {'expansion_factor': 1.0}
        self.input_layer_neuron_count = 7
        self.neuron_signal_size = 6

    def update_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.__dict__:
                setattr(self, key, value)