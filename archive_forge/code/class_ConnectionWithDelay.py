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
class ConnectionWithDelay:

    def __init__(self, strength, repeat_factor, invert_signal):
        self.strength = self._validate_param(strength, 'Strength', float, int, min_val=0, max_val=127)
        self.repeat_factor = self._validate_param(repeat_factor, 'Repeat Factor', int, min_val=1, max_val=5)
        self.invert_signal = self._validate_param(invert_signal, 'Invert Signal', bool)
        self.delayed_signals = deque([0] * self.repeat_factor, maxlen=self.repeat_factor)

    def transmit(self, signal, simulation_params):
        if not isinstance(signal, (int, float)):
            raise TypeError('Signal must be a number')
        if not isinstance(simulation_params, SimulationParameters):
            raise TypeError('Simulation parameters must be an instance of SimulationParameters')
        if self.invert_signal:
            signal = -signal
        repeated_signal = signal * self.strength
        self.delayed_signals.append(repeated_signal)
        return self.delayed_signals.popleft()

    def _validate_param(self, param, name, *types, min_val=None, max_val=None):
        if not isinstance(param, types):
            raise TypeError(f'{name} must be of type {types}')
        if min_val is not None and param < min_val:
            raise ValueError(f'{name} must be at least {min_val}')
        if max_val is not None and param > max_val:
            raise ValueError(f'{name} must not exceed {max_val}')
        return param