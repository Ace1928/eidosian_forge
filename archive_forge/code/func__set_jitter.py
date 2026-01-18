import colorsys  # colour format conversions
from math import log, exp, floor, pi
import random  # for jitter values
def _set_jitter(self, value):
    self._jitter = max(0, min(1, value))