from __future__ import division
import math
import os
import signal
import sys
import time
from .compat import *  # for: any, next
from . import widgets
def _env_size(self):
    """Tries to find the term_width from the environment."""
    return int(os.environ.get('COLUMNS', self._DEFAULT_TERMSIZE)) - 1