import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
class _DummyAxis:
    __name__ = 'dummy'

    def __init__(self, minpos=0):
        self._data_interval = (0, 1)
        self._view_interval = (0, 1)
        self._minpos = minpos

    def get_view_interval(self):
        return self._view_interval

    def set_view_interval(self, vmin, vmax):
        self._view_interval = (vmin, vmax)

    def get_minpos(self):
        return self._minpos

    def get_data_interval(self):
        return self._data_interval

    def set_data_interval(self, vmin, vmax):
        self._data_interval = (vmin, vmax)

    def get_tick_space(self):
        return 9