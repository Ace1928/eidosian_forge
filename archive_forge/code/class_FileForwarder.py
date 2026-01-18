import atexit
import inspect
import multiprocessing.connection
import os
import signal
import subprocess
import sys
import time
import pickle
from ..Qt import QT_LIB, mkQApp
from ..util import cprint  # color printing for debugging
from .remoteproxy import (
import threading
class FileForwarder(threading.Thread):
    """
    Background thread that forwards data from one pipe to another. 
    This is used to catch data from stdout/stderr of the child process
    and print it back out to stdout/stderr. We need this because this
    bug: http://bugs.python.org/issue3905  _requires_ us to catch
    stdout/stderr.

    *output* may be a file or 'stdout' or 'stderr'. In the latter cases,
    sys.stdout/stderr are retrieved once for every line that is output,
    which ensures that the correct behavior is achieved even if 
    sys.stdout/stderr are replaced at runtime.
    """

    def __init__(self, input, output, color):
        threading.Thread.__init__(self)
        self.input = input
        self.output = output
        self.lock = threading.Lock()
        self.daemon = True
        self.color = color
        self.finish = threading.Event()
        self.start()

    def run(self):
        if self.output == 'stdout' and self.color is not False:
            while not self.finish.is_set():
                line = self.input.readline()
                with self.lock:
                    cprint.cout(self.color, line.decode('utf8'), -1)
        elif self.output == 'stderr' and self.color is not False:
            while not self.finish.is_set():
                line = self.input.readline()
                with self.lock:
                    cprint.cerr(self.color, line.decode('utf8'), -1)
        else:
            if isinstance(self.output, str):
                self.output = getattr(sys, self.output)
            while not self.finish.is_set():
                line = self.input.readline()
                with self.lock:
                    self.output.write(line.decode('utf8'))