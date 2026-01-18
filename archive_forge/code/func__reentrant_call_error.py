import os
import signal
import sys
import threading
import warnings
from . import spawn
from . import util
def _reentrant_call_error(self):
    raise ReentrantCallError('Reentrant call into the multiprocessing resource tracker')