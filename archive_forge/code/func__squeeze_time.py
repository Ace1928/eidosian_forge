from __future__ import print_function
import time
import sys
import os
import shutil
import logging
import pprint
from .disk import mkdirp
def _squeeze_time(t):
    """Remove .1s to the time under Windows: this is the time it take to
    stat files. This is needed to make results similar to timings under
    Unix, for tests
    """
    if sys.platform.startswith('win'):
        return max(0, t - 0.1)
    else:
        return t