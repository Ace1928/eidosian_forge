from __future__ import division
import math
import os
import signal
import sys
import time
from .compat import *  # for: any, next
from . import widgets
def _need_update(self):
    """Returns whether the ProgressBar should redraw the line."""
    if self.currval >= self.next_update or self.finished:
        return True
    delta = time.time() - self.last_update_time
    return self._time_sensitive and delta > self.poll