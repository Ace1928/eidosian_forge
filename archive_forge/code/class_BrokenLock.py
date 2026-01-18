import errno
import fcntl
import multiprocessing
import os
import shutil
import signal
import tempfile
import threading
import time
from fasteners import process_lock as pl
from fasteners import test
class BrokenLock(pl.InterProcessLock):

    def __init__(self, name, errno_code):
        super(BrokenLock, self).__init__(name)
        self.errno_code = errno_code

    def unlock(self):
        pass

    def trylock(self):
        err = IOError()
        err.errno = self.errno_code
        raise err