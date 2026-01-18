import os
import signal
import sys
import threading
import warnings
from . import spawn
from . import util
def _check_alive(self):
    """Check that the pipe has not been closed by sending a probe."""
    try:
        os.write(self._fd, b'PROBE:0:noop\n')
    except OSError:
        return False
    else:
        return True