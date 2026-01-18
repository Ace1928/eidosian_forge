import errno
import logging
import os
import threading
import time
import six
from fasteners import _utils
def _do_close(self):
    if self.lockfile is not None:
        self.lockfile.close()
        self.lockfile = None