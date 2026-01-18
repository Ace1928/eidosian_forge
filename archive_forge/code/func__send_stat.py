import random
from collections import deque
from datetime import timedelta
from .timer import Timer
def _send_stat(self, stat, value, rate):
    self._after(self._prepare(stat, value, rate))