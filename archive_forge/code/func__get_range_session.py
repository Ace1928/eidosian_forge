import atexit
import datetime
import re
import sqlite3
import threading
from pathlib import Path
from decorator import decorator
from traitlets import (
from traitlets.config.configurable import LoggingConfigurable
from IPython.paths import locate_profile
from IPython.utils.decorators import undoc
def _get_range_session(self, start=1, stop=None, raw=True, output=False):
    """Get input and output history from the current session. Called by
        get_range, and takes similar parameters."""
    input_hist = self.input_hist_raw if raw else self.input_hist_parsed
    n = len(input_hist)
    if start < 0:
        start += n
    if not stop or stop > n:
        stop = n
    elif stop < 0:
        stop += n
    for i in range(start, stop):
        if output:
            line = (input_hist[i], self.output_hist_reprs.get(i))
        else:
            line = input_hist[i]
        yield (0, i, line)